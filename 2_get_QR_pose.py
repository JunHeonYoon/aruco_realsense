import argparse
import json
import csv
import os
import numpy as np
import cv2
import pyrealsense2 as rs
import yaml

def load_camera_calibration(calib_path):
    """Load camera_matrix and dist_coeffs from a YAML file."""
    with open(calib_path, 'r') as f:
        data = yaml.safe_load(f)
    cm = np.array(data['camera_matrix'], dtype=np.float32)
    dc = np.array(data['dist_coeff'],    dtype=np.float32)
    return cm, dc

def init_realsense(width, height, fps):
    """Initialize and start a RealSense color-only pipeline."""
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
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

def save_poses(poses, output_path):
    """Save the list of poses to JSON or CSV, based on file extension."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.json':
        with open(output_path, 'w') as f:
            json.dump(poses, f, indent=2)
    elif ext == '.csv':
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id','r11','r12','r13','tx',
                             'r21','r22','r23','ty',
                             'r31','r32','r33','tz'])
            for p in poses:
                T = np.array(p['tf'])
                row = [p['id']] + T[:3, :].flatten().tolist()
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

    # 1) Load or defer calibration
    if args.calib:
        camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
    else:
        camera_matrix = dist_coeffs = None

    # 2) Prepare ArUco detection
    aruco_dict   = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, 'DICT_' + args.dictionary))
    aruco_params = cv2.aruco.DetectorParameters()

    # 3) Start RealSense pipeline
    pipeline = init_realsense(args.width, args.height, args.fps)

    if not args.no_display:
        cv2.namedWindow('ArUco', cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img_rgb = np.asanyarray(color.get_data())
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Auto-fetch intrinsics if needed
            if camera_matrix is None:
                prof = pipeline.get_active_profile()
                intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                camera_matrix = np.array([[intr.fx, 0,       intr.ppx],
                                          [0,       intr.fy, intr.ppy],
                                          [0,       0,       1       ]], dtype=np.float32)
                dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

            # Detect and draw poses
            poses = detect_aruco_poses(img_bgr, camera_matrix, dist_coeffs,
                                       aruco_dict, aruco_params,
                                       args.marker_length,
                                       draw=not args.no_display)

            if poses:
                print(json.dumps(poses, indent=2))

            if not args.no_display:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.imshow('ArUco', img_bgr)

    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        if args.output and poses:
            save_poses(poses, args.output)
            print(f"Saved poses to {args.output}")

if __name__ == "__main__":
    main()
