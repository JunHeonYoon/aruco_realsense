#!/usr/bin/env python3
"""
Detect an ArUco marker with a RealSense camera and publish
<base_frame> -> <qr_frame> as a TF in real time.

Key:
    q : quit node
"""

import argparse
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation


def load_intrinsics(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    camera_matrix = np.array(y["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(y["dist_coeff"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def overlay_help(img, text, alpha=0.6):
    """Draw translucent banner with instructions."""
    banner = img.copy()
    h, w = img.shape[:2]
    pad, y0 = 10, 30
    cv2.rectangle(banner, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(
        banner,
        text,
        (pad, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.addWeighted(banner, alpha, img, 1 - alpha, 0, img)


class PoseFilter:
    def __init__(self, window=10):
        self.buf = deque(maxlen=window)

    def add(self, pos, quat):
        self.buf.append((pos, quat))

    def get(self):
        if not self.buf:
            return None
        pos = np.mean([b[0] for b in self.buf], axis=0)
        quat = np.mean([b[1] for b in self.buf], axis=0)
        norm = np.linalg.norm(quat)
        if norm > 1e-9:
            quat /= norm
        return pos, quat


class QRPublisher(Node):
    def __init__(self, args):
        super().__init__("qr_tf_publisher")
        self.args = args

        self.tfb = tf2_ros.TransformBroadcaster(self)
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        self.pipe.start(cfg)

        if args.intrinsics:
            self.K, self.dist = load_intrinsics(args.intrinsics)
            self.get_logger().info(f"Loaded intrinsics from {args.intrinsics}")
        else:
            self.K = None
            self.dist = None

        self.dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_" + args.dictionary))
        self.par = cv2.aruco.DetectorParameters()
        self.mlen = args.marker_length

        self.base = args.base_frame
        self.cam = args.cam_frame
        self.qr = args.qr_frame
        self.filt = PoseFilter(args.filter_N)

        self.gui = not args.no_display
        if self.gui:
            cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    def spin(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
                base2cam = self.lookup_tf(self.base, self.cam)
                frame = self.pipe.wait_for_frames().get_color_frame()
                if not frame:
                    continue
                img = np.asanyarray(frame.get_data()).copy()

                cam2qr = self.detect_marker(img)
                if base2cam is not None and cam2qr is not None:
                    base2qr = base2cam @ cam2qr
                    pos = base2qr[:3, 3]
                    quat = Rotation.from_matrix(base2qr[:3, :3]).as_quat()
                    self.filt.add(pos, quat)
                    out = self.filt.get()
                    if out:
                        self.publish_tf(out[0], out[1])

                if self.gui:
                    overlay_help(img, "q: quit")
                    cv2.imshow("Live", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.pipe.stop()
            cv2.destroyAllWindows()

    def lookup_tf(self, parent, child):
        try:
            ts = self.buf.lookup_transform(
                parent,
                child,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception:
            return None

        transform = np.eye(4)
        t = ts.transform.translation
        q = ts.transform.rotation
        transform[:3, 3] = [t.x, t.y, t.z]
        transform[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return transform

    def detect_marker(self, img_bgr):
        if self.K is None:
            intr = self.pipe.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array(
                [
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            self.dist = np.array(intr.coeffs[:5], dtype=np.float32)
            self.get_logger().info("Intrinsics auto-fetched from RealSense")

        corners, ids, _ = cv2.aruco.detectMarkers(img_bgr, self.dict, parameters=self.par)
        if ids is None:
            return None

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0],
            self.mlen,
            self.K,
            self.dist,
        )
        cv2.aruco.drawDetectedMarkers(img_bgr, [corners[0]])
        cv2.drawFrameAxes(img_bgr, self.K, self.dist, rvec[0], tvec[0], self.mlen * 0.5)

        rotation_matrix, _ = cv2.Rodrigues(rvec[0])
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = tvec[0].flatten()
        return transform

    def publish_tf(self, pos, quat):
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.base
        ts.child_frame_id = self.qr
        ts.transform.translation.x = float(pos[0])
        ts.transform.translation.y = float(pos[1])
        ts.transform.translation.z = float(pos[2])
        ts.transform.rotation.x = float(quat[0])
        ts.transform.rotation.y = float(quat[1])
        ts.transform.rotation.z = float(quat[2])
        ts.transform.rotation.w = float(quat[3])
        self.tfb.sendTransform(ts)


def main():
    ap = argparse.ArgumentParser(description="Publish base->QR TF using RealSense & ArUco (ROS 2)")
    ap.add_argument("--intrinsics", default="camIntrinsic.yaml", help="YAML with camera_matrix & dist_coeff")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--marker-length", type=float, default=0.04)
    ap.add_argument(
        "--dictionary",
        choices=["4X4_50", "5X5_100", "6X6_250", "7X7_1000"],
        default="6X6_250",
    )
    ap.add_argument("--base-frame", default="panda_hand")
    ap.add_argument("--cam-frame", default="rs_camera")
    ap.add_argument("--qr-frame", default="object_frame")
    ap.add_argument(
        "--filter-N",
        type=int,
        default=10,
        help="Window size for moving-average filter",
    )
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    rclpy.init(args=None)
    node = QRPublisher(args)
    try:
        node.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
