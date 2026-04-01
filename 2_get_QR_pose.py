import argparse
import csv
import json
import os

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from pyk4a import (
        FPS as K4AFPS,
        CalibrationType,
        ColorResolution,
        Config as K4AConfig,
        DepthMode,
        PyK4A,
        connected_device_count,
    )
except ImportError:
    K4AFPS = None
    CalibrationType = None
    ColorResolution = None
    K4AConfig = None
    DepthMode = None
    PyK4A = None
    connected_device_count = None


AZURE_KINECT_RESOLUTION_MAP = {
    (1280, 720): ColorResolution.RES_720P if ColorResolution else None,
    (1920, 1080): ColorResolution.RES_1080P if ColorResolution else None,
    (2560, 1440): ColorResolution.RES_1440P if ColorResolution else None,
    (2048, 1536): ColorResolution.RES_1536P if ColorResolution else None,
    (3840, 2160): ColorResolution.RES_2160P if ColorResolution else None,
    (4096, 3072): ColorResolution.RES_3072P if ColorResolution else None,
}

AZURE_KINECT_FPS_MAP = {
    5: K4AFPS.FPS_5 if K4AFPS else None,
    15: K4AFPS.FPS_15 if K4AFPS else None,
    30: K4AFPS.FPS_30 if K4AFPS else None,
}


def require_backend(backend):
    if backend == "azure_kinect" and PyK4A is None:
        raise ImportError(
            "Azure Kinect backend requires 'pyk4a'. Install Azure Kinect SDK and run "
            "'pip install pyk4a'."
        )
    if backend == "realsense" and rs is None:
        raise ImportError(
            "RealSense backend requires 'pyrealsense2'. Run 'pip install pyrealsense2'."
        )


def list_realsense_devices():
    """Return connected RealSense devices with stable identifiers."""
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        devices.append(
            {
                "serial": serial,
                "name": name,
            }
        )
    return devices


def list_azure_kinect_devices():
    """Return connected Azure Kinect devices with stable identifiers."""
    count = connected_device_count()
    devices = []
    for device_id in range(count):
        camera = PyK4A(device_id=device_id)
        camera.open()
        try:
            devices.append(
                {
                    "device_id": device_id,
                    "serial": camera.serial,
                    "name": f"Azure Kinect DK #{device_id}",
                }
            )
        finally:
            camera.close()
    return devices


def resolve_calibration_path(calib_path, serial):
    """
    Resolve calibration source for one camera.
    If calib_path is a directory, prefer <serial>.yaml or <serial>.yml.
    """
    if not calib_path:
        return None

    if os.path.isdir(calib_path):
        for ext in (".yaml", ".yml"):
            candidate = os.path.join(calib_path, f"{serial}{ext}")
            if os.path.isfile(candidate):
                return candidate
        return None

    return calib_path


def load_camera_calibration(calib_path):
    with open(calib_path, "r") as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeff"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def load_realsense_calibration(calib_path, pipeline):
    """Load camera intrinsics from YAML or from an active RealSense pipeline."""
    if calib_path and os.path.isfile(calib_path):
        return load_camera_calibration(calib_path)

    prof = pipeline.get_active_profile()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array(
        [
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)
    print("Fetched intrinsics from RealSense.")
    return camera_matrix, dist_coeffs


def load_azure_kinect_calibration(calib_path, camera):
    """Load camera intrinsics from YAML or from Azure Kinect factory calibration."""
    if calib_path and os.path.isfile(calib_path):
        return load_camera_calibration(calib_path)

    calibration = camera.calibration
    camera_matrix = calibration.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
    dist_coeffs = calibration.get_distortion_coefficients(CalibrationType.COLOR).astype(np.float32)
    print("Fetched intrinsics from Azure Kinect.")
    return camera_matrix, dist_coeffs


def init_realsense(serial, width, height, fps):
    """Initialize and start a RealSense color-only pipeline for one device."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def init_azure_kinect(device_id, width, height, fps):
    """Initialize and start an Azure Kinect color-only stream for one device."""
    resolution = AZURE_KINECT_RESOLUTION_MAP.get((width, height))
    fps_enum = AZURE_KINECT_FPS_MAP.get(fps)
    if resolution is None:
        supported = ", ".join(f"{w}x{h}" for w, h in AZURE_KINECT_RESOLUTION_MAP)
        raise ValueError(
            f"Azure Kinect color resolution {width}x{height} is not supported. "
            f"Supported values: {supported}"
        )
    if fps_enum is None:
        raise ValueError("Azure Kinect supports only 5, 15, or 30 FPS.")

    camera = PyK4A(
        config=K4AConfig(
            color_resolution=resolution,
            camera_fps=fps_enum,
            depth_mode=DepthMode.OFF,
            synchronized_images_only=False,
        ),
        device_id=device_id,
    )
    camera.start()
    return camera


def get_realsense_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color:
        return None
    return np.asanyarray(color.get_data()).copy()


def get_azure_kinect_frame(camera):
    capture = camera.get_capture()
    color = capture.color
    if color is None:
        return None

    if color.ndim == 3 and color.shape[2] == 4:
        return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    return color.copy()


def detect_aruco_poses(
    frame_bgr,
    camera_matrix,
    dist_coeffs,
    aruco_dict,
    aruco_params,
    marker_length,
    draw=True,
):
    """
    Detect ArUco markers in a BGR image and return pose dictionaries.
    """
    corners, ids, _ = cv2.aruco.detectMarkers(
        frame_bgr,
        aruco_dict,
        parameters=aruco_params,
    )
    poses = []
    if ids is None:
        return poses

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs,
    )

    for i, marker_id in enumerate(ids.flatten()):
        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = tvecs[i].flatten()
        poses.append({"id": int(marker_id), "tf": transform.tolist()})

        if draw:
            cv2.aruco.drawDetectedMarkers(frame_bgr, [corners[i]])
            cv2.drawFrameAxes(
                frame_bgr,
                camera_matrix,
                dist_coeffs,
                rvecs[i],
                tvecs[i],
                marker_length * 0.5,
            )
            text_pt = tuple(corners[i][0][0].astype(int))
            cv2.putText(
                frame_bgr,
                f"ID:{marker_id}",
                (text_pt[0], text_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return poses


def annotate_camera_frame(frame_bgr, camera_name, serial, poses):
    """Add camera-level status text to a display frame."""
    lines = [
        camera_name,
        f"Serial: {serial}",
        f"Markers: {len(poses)}",
    ]
    y = 30
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y += 28


def save_poses(poses, output_path):
    """Save the list of poses to JSON or CSV, based on file extension."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".json":
        with open(output_path, "w") as f:
            json.dump(poses, f, indent=2)
    elif ext == ".csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "camera_serial",
                    "camera_name",
                    "id",
                    "r11",
                    "r12",
                    "r13",
                    "tx",
                    "r21",
                    "r22",
                    "r23",
                    "ty",
                    "r31",
                    "r32",
                    "r33",
                    "tz",
                ]
            )
            for camera_entry in poses:
                for pose in camera_entry["poses"]:
                    transform = np.array(pose["tf"])
                    row = [
                        camera_entry["camera_serial"],
                        camera_entry["camera_name"],
                        pose["id"],
                    ] + transform[:3, :].flatten().tolist()
                    writer.writerow(row)
    else:
        raise ValueError("Unsupported output format. Use .json or .csv")


def build_camera_states(args):
    if args.backend == "azure_kinect":
        devices = list_azure_kinect_devices()
        if not devices:
            raise RuntimeError("No Azure Kinect devices found.")
    else:
        devices = list_realsense_devices()
        if not devices:
            raise RuntimeError("No RealSense devices found.")

    camera_states = []
    for device in devices:
        if args.backend == "azure_kinect":
            camera = init_azure_kinect(device["device_id"], args.width, args.height, args.fps)
            calib_path = resolve_calibration_path(args.calib, device["serial"])
            camera_matrix, dist_coeffs = load_azure_kinect_calibration(calib_path, camera)
        else:
            camera = init_realsense(device["serial"], args.width, args.height, args.fps)
            calib_path = resolve_calibration_path(args.calib, device["serial"])
            camera_matrix, dist_coeffs = load_realsense_calibration(calib_path, camera)

        window_name = f"ArUco - {device['serial']}"
        camera_states.append(
            {
                "backend": args.backend,
                "serial": device["serial"],
                "name": device["name"],
                "camera": camera,
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "window_name": window_name,
            }
        )

        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    return camera_states


def stop_camera(state):
    if state["backend"] == "azure_kinect":
        state["camera"].stop()
    else:
        state["camera"].stop()


def fetch_frame(state):
    if state["backend"] == "azure_kinect":
        return get_azure_kinect_frame(state["camera"])
    return get_realsense_frame(state["camera"])


def main():
    parser = argparse.ArgumentParser(description="Azure Kinect / RealSense + ArUco Pose Estimator")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["azure_kinect", "realsense"],
        default="azure_kinect",
        help="Camera backend to use",
    )
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=720, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="YAML camera calibration file or directory",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.05,
        help="Marker side length in meters",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        choices=["4X4_50", "5X5_100", "6X6_250", "7X7_1000"],
        default="6X6_250",
        help="Predefined ArUco dictionary",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window")
    parser.add_argument("--output", type=str, help="Save poses to .json or .csv")
    args = parser.parse_args()

    require_backend(args.backend)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_" + args.dictionary))
    aruco_params = cv2.aruco.DetectorParameters()
    camera_states = build_camera_states(args)

    print(f"Started {len(camera_states)} {args.backend} camera(s):")
    for state in camera_states:
        print(f"  - {state['name']} [{state['serial']}]")

    last_all_poses = []

    try:
        while True:
            all_poses = []

            for state in camera_states:
                img_bgr = fetch_frame(state)
                if img_bgr is None:
                    continue

                poses = detect_aruco_poses(
                    img_bgr,
                    state["camera_matrix"],
                    state["dist_coeffs"],
                    aruco_dict,
                    aruco_params,
                    args.marker_length,
                    draw=not args.no_display,
                )

                camera_entry = {
                    "camera_serial": state["serial"],
                    "camera_name": state["name"],
                    "poses": poses,
                }
                all_poses.append(camera_entry)

                if not args.no_display:
                    annotate_camera_frame(img_bgr, state["name"], state["serial"], poses)
                    cv2.imshow(state["window_name"], img_bgr)

            last_all_poses = all_poses
            cameras_with_poses = [entry for entry in all_poses if entry["poses"]]
            if cameras_with_poses:
                print(json.dumps(cameras_with_poses, indent=2))

            if not args.no_display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    finally:
        for state in camera_states:
            stop_camera(state)
        if not args.no_display:
            cv2.destroyAllWindows()
        if args.output and last_all_poses:
            save_poses(last_all_poses, args.output)
            print(f"Saved poses to {args.output}")


if __name__ == "__main__":
    main()
