import argparse
import csv
import json
import os
import numpy as np
import cv2
import pyrealsense2 as rs
import yaml

def list_realsense_devices():
    """Return connected RealSense devices with stable identifiers."""
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        devices.append({
            "serial": serial,
            "name": name,
        })
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

def load_camera_calibration(calib_path, pipeline):
    """Load camera_matrix and dist_coeffs from a YAML file."""
    if calib_path and os.path.isfile(calib_path):
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        cm = np.array(data['camera_matrix'], dtype=np.float32)
        dc = np.array(data['dist_coeff'],    dtype=np.float32)
    else:
        prof = pipeline.get_active_profile()
        intr = prof.get_stream(rs.stream.color
                ).as_video_stream_profile().get_intrinsics()
        cm = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float32)
        dc = np.array(intr.coeffs[:5], dtype=np.float32)
        print("Fetched intrinsics from RealSense.")
    return cm, dc

def init_realsense(serial, width, height, fps):
    """Initialize and start a RealSense color-only pipeline for one device."""
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline

def detect_aruco_poses(frame_bgr, camera_matrix, dist_coeffs,
                       aruco_dict, aruco_params, marker_length, draw=True):
    """
    Detects ArUco markers in a BGR image, estimates their poses,
    and returns a list of {id, tf} dictionaries (4x4 lists).
    """
    corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, aruco_dict, parameters=aruco_params)
    poses = []
    if ids is None:
        return poses

    # Estimate rvecs and tvecs for each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs)

    for i, marker_id in enumerate(ids.flatten()):
        # Build the 4×4 transformation matrix
        R, _ = cv2.Rodrigues(rvecs[i])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3,  3] = tvecs[i].flatten()
        poses.append({"id": int(marker_id), "tf": T.tolist()})

        if draw:
            cv2.aruco.drawDetectedMarkers(frame_bgr, [corners[i]])
            cv2.drawFrameAxes(frame_bgr, camera_matrix, dist_coeffs,
                              rvecs[i], tvecs[i], marker_length * 0.5)
            # Put the marker ID above its top-left corner
            text_pt = tuple(corners[i][0][0].astype(int))
            cv2.putText(frame_bgr, f"ID:{marker_id}",
                        (text_pt[0], text_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return poses

def annotate_camera_frame(frame_bgr, camera_name, serial, poses):
    """Add camera-level status text to a display frame."""
    lines = [
        f"{camera_name}",
        f"Serial: {serial}",
        f"Markers: {len(poses)}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y += 28

def save_poses(poses, output_path):
    """Save the list of poses to JSON or CSV, based on file extension."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.json':
        with open(output_path, 'w') as f:
            json.dump(poses, f, indent=2)
    elif ext == '.csv':
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['camera_serial', 'camera_name', 'id',
                             'r11','r12','r13','tx',
                             'r21','r22','r23','ty',
                             'r31','r32','r33','tz'])
            for camera_entry in poses:
                for p in camera_entry['poses']:
                    T = np.array(p['tf'])
                    row = [camera_entry['camera_serial'],
                           camera_entry['camera_name'],
                           p['id']] + T[:3, :].flatten().tolist()
                    writer.writerow(row)
    else:
        raise ValueError("Unsupported output format. Use .json or .csv")

def main():
    parser = argparse.ArgumentParser(description="RealSense + ArUco Pose Estimator")
    parser.add_argument('--width',        type=int,   default=1280, help="Color stream width")
    parser.add_argument('--height',       type=int,   default=720,  help="Color stream height")
    parser.add_argument('--fps',          type=int,   default=30,   help="Color stream FPS")
    parser.add_argument('--calib',        type=str,   default="camIntrinsic.yaml", help="YAML camera calibration file")
    parser.add_argument('--marker-length',type=float, default=0.05,help="Marker side length in meters")
    parser.add_argument('--dictionary',   type=str,
                        choices=['4X4_50','5X5_100','6X6_250','7X7_1000'],
                        default='6X6_250', help="Predefined ArUco dictionary")
    parser.add_argument('--no-display',   action='store_true', help="Disable OpenCV window")
    parser.add_argument('--output',       type=str,             help="Save poses to .json or .csv")
    args = parser.parse_args()

    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense devices found.")

    # 1) Prepare ArUco detection
    aruco_dict   = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, 'DICT_' + args.dictionary))
    aruco_params = cv2.aruco.DetectorParameters()

    camera_states = []
    for device in devices:
        pipeline = init_realsense(device["serial"], args.width, args.height, args.fps)
        calib_path = resolve_calibration_path(args.calib, device["serial"])

        if args.calib:
            camera_matrix, dist_coeffs = load_camera_calibration(calib_path, pipeline)
        else:
            camera_matrix = dist_coeffs = None

        window_name = f"ArUco - {device['serial']}"
        camera_states.append({
            "serial": device["serial"],
            "name": device["name"],
            "pipeline": pipeline,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "window_name": window_name,
        })

        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Started {len(camera_states)} RealSense camera(s):")
    for state in camera_states:
        print(f"  - {state['name']} [{state['serial']}]")

    last_all_poses = []

    try:
        while True:
            all_poses = []

            for state in camera_states:
                frames = state["pipeline"].wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue

                img_bgr = np.asanyarray(color.get_data()).copy()

                # Auto-fetch intrinsics if needed
                if state["camera_matrix"] is None:
                    prof = state["pipeline"].get_active_profile()
                    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    state["camera_matrix"] = np.array([[intr.fx, 0,       intr.ppx],
                                                       [0,       intr.fy, intr.ppy],
                                                       [0,       0,       1       ]], dtype=np.float32)
                    state["dist_coeffs"] = np.array(intr.coeffs[:5], dtype=np.float32)

                poses = detect_aruco_poses(img_bgr,
                                           state["camera_matrix"],
                                           state["dist_coeffs"],
                                           aruco_dict,
                                           aruco_params,
                                           args.marker_length,
                                           draw=not args.no_display)

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

            if not args.no_display and (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    finally:
        for state in camera_states:
            state["pipeline"].stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        if args.output and last_all_poses:
            save_poses(last_all_poses, args.output)
            print(f"Saved poses to {args.output}")

if __name__ == "__main__":
    main()

'''
If your OpenCV version is lower than 4.7, please upgrade by:
pip install --upgrade opencv-contrib-python
'''