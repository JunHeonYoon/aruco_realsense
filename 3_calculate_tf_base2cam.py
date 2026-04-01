"""
Hand-Eye TF Collector with Azure Kinect & ArUco
    s : sample TF pair
    c : calibrate + save YAML
    q : quit
"""

import argparse
import yaml
import sys
import os
import numpy as np
import cv2
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from numpy.linalg import inv
from scipy.linalg import sqrtm
from math import atan2, asin

from azure_kinect_camera import (
    get_bgr_frame,
    get_camera_intrinsics,
    select_azure_kinect_device,
    start_azure_kinect,
)


# ------------------------------------------------
# Utility – draw a translucent help banner
# ------------------------------------------------
def overlay_help(img, lines, alpha=0.6):
    """
    img   : BGR image (modified in-place)
    lines : list of strings to draw
    alpha : overlay opacity
    """
    banner = img.copy()
    h, w = img.shape[:2]
    pad, y0 = 10, 30
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2

    # dark rectangle across the top
    cv2.rectangle(banner, (0, 0),
                  (w, y0 + 25 * len(lines)), (0, 0, 0), -1)

    for i, text in enumerate(lines):
        cv2.putText(banner, text, (pad, y0 + i * 25),
                    font, scale, (0, 255, 0), thick, cv2.LINE_AA)

    cv2.addWeighted(banner, alpha, img, 1 - alpha, 0, img)


# ------------------------------------------------
# Basic math helpers
# ------------------------------------------------
def load_yaml_intrinsics(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return np.array(d['camera_matrix'], dtype=np.float32), \
           np.array(d['dist_coeff'],    dtype=np.float32)


def rot_to_quat(R):
    """3x3 -> quaternion [x,y,z,w]."""
    q = np.empty(4, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        j, k = (1, 2, 0)[i], (2, 0, 1)[i]
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[i] = 0.25 * s
        q[3] = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return q.tolist()


def rot_to_euler_zyx(R):
    """3x3 -> [roll, pitch, yaw] (ZYX)."""
    if abs(R[2, 0]) < 1.0:
        pitch = asin(-R[2, 0])
        roll  = atan2(R[2, 1], R[2, 2])
        yaw   = atan2(R[1, 0], R[0, 0])
    else:  # gimbal-lock
        pitch = np.pi / 2 * np.sign(-R[2, 0])
        roll  = atan2(-R[0, 1], R[1, 1])
        yaw   = 0.0
    return [float(roll), float(pitch), float(yaw)]


def logR(T):
    R = T[:3, :3]
    th = np.arccos((np.trace(R) - 1) / 2)
    if abs(np.sin(th)) < 1e-6:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * th / (2 * np.sin(th))


# --------------------------
# Calibration (AX = XB)
# --------------------------
def solve_AX_XB(A_list, B_list):
    n_data = len(A_list)
    M = np.zeros((3,3))
    C = np.zeros((3*n_data, 3))
    d = np.zeros((3*n_data, 1))

    # Build M
    for i in range(n_data):
        alpha = logR(A_list[i])
        beta = logR(B_list[i])
        M += np.outer(beta, alpha)

    # Solve for rotation
    try:
        M_inv = inv(M.T @ M)
        theta = sqrtm(M_inv) @ M.T
    except np.linalg.LinAlgError:
        print("Error: Cannot invert M matrix for rotation solution.")
        return None, None

    # Solve for translation
    for i in range(n_data):
        rot_a = A_list[i][0:3, 0:3]
        trans_a = A_list[i][0:3, 3]
        trans_b = B_list[i][0:3, 3]

        C[3*i:3*i+3, :] = np.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - theta @ trans_b

    try:
        b_x = inv(C.T @ C) @ (C.T @ d)
    except np.linalg.LinAlgError:
        print("Error: Cannot invert C matrix for translation solution.")
        return None, None

    return theta, b_x


# ------------------------------------------------
# Main collector class
# ------------------------------------------------
class TFCollector:
    def __init__(self, cfg):
        self.cfg = cfg

        # ROS 2
        rclpy.init(args=None)
        self.node = Node('tf_collector_node')
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self.node)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        self.tf_pub = self.node.create_publisher(
            TransformStamped, 'calibrated_tf', 10)
        self.pose_pub = self.node.create_publisher(PoseStamped, 'cam_pose', 10)

        # Azure Kinect
        self.device = select_azure_kinect_device(cfg.camera_serial)
        self.camera = start_azure_kinect(self.device['device_id'], cfg.width, cfg.height, cfg.fps)
        print(f"Started Azure Kinect stream from [{self.device['serial']}]")

        # Intrinsics
        if cfg.intrinsics and os.path.isfile(cfg.intrinsics):
            self.K, self.dist = load_yaml_intrinsics(cfg.intrinsics)
            print(f"Loaded intrinsics from {cfg.intrinsics}")
        else:
            self.K = self.dist = None
            if cfg.intrinsics:
                print(f"Intrinsics file not found: {cfg.intrinsics}")
            print("Using Azure Kinect factory intrinsics from the camera.")

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, 'DICT_' + cfg.dictionary))
        self.aruco_param = cv2.aruco.DetectorParameters()

        # Sample buffers
        self.list_tf_base2ee = []
        self.list_tf_cam2qr = []

        # OpenCV windows
        if not cfg.no_display:
            cv2.namedWindow('Live View', cv2.WINDOW_NORMAL)
            cv2.namedWindow('TF Info',  cv2.WINDOW_NORMAL)

    # --------------------------
    # Real-time helpers
    # --------------------------
    def get_intrinsics_if_needed(self):
        if self.K is not None:
            return
        self.K, self.dist = get_camera_intrinsics(self.camera)
        print("Fetched intrinsics from Azure Kinect.")

    def tf_to_mat(self, tf_msg):
        T = np.eye(4)
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                      [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                      [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def get_base2ee(self):
        try:
            tfm = self.tf_buf.lookup_transform(
                self.cfg.base_frame,
                self.cfg.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            return self.tf_to_mat(tfm)
        except Exception:
            return None

    def get_cam2qr(self, img):
        self.get_intrinsics_if_needed()
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict,
                                                  parameters=self.aruco_param)
        if ids is None:
            return None
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], self.cfg.marker_length, self.K, self.dist)
        cv2.aruco.drawDetectedMarkers(img, [corners[0]])
        cv2.drawFrameAxes(img, self.K, self.dist,
                          rvec[0], tvec[0], self.cfg.marker_length * 0.5)
        R, _ = cv2.Rodrigues(rvec[0])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tvec[0].flatten()
        return T

    # --------------------------
    # Main loop
    # --------------------------
    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            base2ee = self.get_base2ee()

            img = get_bgr_frame(self.camera)
            if img is None:
                continue

            cam2qr = self.get_cam2qr(img)

            # Info image
            info = np.ones((300, 600, 3), dtype=np.uint8) * 255
            y = 20
            font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
            for title, M in (("Base->EE", base2ee), ("Cam->QR", cam2qr)):
                cv2.putText(info, f"{title}:", (10, y), font, scale, (0, 0, 0), 1)
                y += 18
                if M is not None:
                    for row in M[:3]:
                        cv2.putText(info, " ".join(f"{v:.3f}" for v in row),
                                    (10, y), font, scale, (0, 0, 0), 1)
                        y += 18
                else:
                    cv2.putText(info, "None", (10, y), font, scale, (0, 0, 0), 1)
                    y += 18
                y += 10

            # Overlay help
            overlay_help(img, ["Keys :  s = Sample    c = Calibrate & Save    q = Quit"])

            # Show
            if not self.cfg.no_display:
                cv2.imshow('Live View', img)
                cv2.imshow('TF Info', info)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if base2ee is not None and cam2qr is not None:
                    self.list_tf_base2ee.append(base2ee.copy())
                    self.list_tf_cam2qr.append(cam2qr.copy())
                    print(f"Sample #{len(self.list_tf_base2ee)} recorded.")
                else:
                    print("Cannot sample - missing TF.")
            elif key == ord('c'):
                if len(self.list_tf_base2ee) < 2:
                    print("Need at least 2 samples.")
                    continue
                A_list = []
                B_list = []
                p = len(self.list_tf_base2ee)
                for i in range(p - 1):
                    # This usage might vary depending on how you interpret your hand–eye setup
                    # The code from your original snippet used (inv for base->EE?), adapt if needed
                    A = self.list_tf_base2ee[i + 1] @ inv(self.list_tf_base2ee[i])
                    B = self.list_tf_cam2qr[i + 1] @ inv(self.list_tf_cam2qr[i])
                    A_list.append(A)
                    B_list.append(B)
                R, t = solve_AX_XB(A_list, B_list)
                X = np.eye(4); X[:3, :3] = R.real; X[:3, 3] = t.flatten()
                t_xyz = [float(v) for v in X[:3, 3]]

                # Compose YAML
                data = {
                    'rotation_matrix': X[:3, :3].tolist(),
                    'quaternion':      rot_to_quat(X[:3, :3]),
                    'euler_zyx':       rot_to_euler_zyx(X[:3, :3]),
                    'translation':     t_xyz
                }
                with open(self.cfg.output, 'w') as f:
                    yaml.dump(data, f)
                print(f"Saved calibration to {self.cfg.output}")

                # Publish TF and pose through ROS 2
                tfm = TransformStamped()
                tfm.header.stamp = self.node.get_clock().now().to_msg()
                tfm.header.frame_id = self.cfg.calib_frame
                tfm.child_frame_id  = self.cfg.cam_frame
                tfm.transform.translation.x = t_xyz[0]
                tfm.transform.translation.y = t_xyz[1]
                tfm.transform.translation.z = t_xyz[2]
                q = data['quaternion']
                tfm.transform.rotation.x, tfm.transform.rotation.y, tfm.transform.rotation.z, tfm.transform.rotation.w = q
                self.tf_pub.publish(tfm)
                self.tf_broadcaster.sendTransform(tfm)

                # PoseStamped
                pose = PoseStamped()
                pose.header = tfm.header
                pose.pose.position.x = t_xyz[0]
                pose.pose.position.y = t_xyz[1]
                pose.pose.position.z = t_xyz[2]
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
                self.pose_pub.publish(pose)
                print("Calibration published.")
            elif key == ord('q'):
                break

        self.camera.stop()
        cv2.destroyAllWindows()
        self.node.destroy_node()
        rclpy.shutdown()


# ------------------------------------------------
# CLI
# ------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Azure Kinect ArUco Hand-Eye Collector")
    p.add_argument('--intrinsics', default=None, help='Optional YAML with camera_matrix & dist_coeff')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--camera-serial', default=None,
                   help='Azure Kinect serial number to use. If omitted or "none", use the first detected camera.')
    p.add_argument('--base-frame', default='mobile_base')
    p.add_argument('--ee-frame',   default='right_fr3_hand')
    p.add_argument('--cam-frame',  default='camera')
    p.add_argument('--calib-frame', default='camera_calibrated')
    p.add_argument('--marker-length', type=float, default=0.04)
    p.add_argument('--dictionary',
                   choices=['4X4_50', '5X5_100', '6X6_250', '7X7_1000'],
                   default='6X6_250')
    p.add_argument('--output', default='TF_base2cam.yaml', help='Output YAML file')
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    TFCollector(cfg).run()
