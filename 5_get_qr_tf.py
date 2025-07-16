#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect an ArUco marker with a RealSense camera and publish
<base_frame> → <qr_frame> as a TF in real time.

Key:
    q : quit node
"""

import argparse
from collections import deque
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import TransformStamped
from math import sqrt, atan2, asin


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_intrinsics(path):
    with open(path, 'r') as f:
        y = yaml.safe_load(f)
    K   = np.array(y['camera_matrix'], dtype=np.float32)
    dst = np.array(y['dist_coeff'],    dtype=np.float32)
    return K, dst


def overlay_help(img, text, alpha=0.6):
    """Draw translucent banner with instructions."""
    banner = img.copy()
    h, w = img.shape[:2]
    pad, y0 = 10, 30
    cv2.rectangle(banner, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(banner, text, (pad, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.addWeighted(banner, alpha, img, 1-alpha, 0, img)


# ------------------------------------------------------------
# Simple moving‑average filter
# ------------------------------------------------------------
class PoseFilter:
    def __init__(self, window=10):
        self.buf = deque(maxlen=window)

    def add(self, p, q):
        self.buf.append((p, q))

    def get(self):
        if not self.buf:
            return None
        pos = np.mean([b[0] for b in self.buf], axis=0)
        quat = np.mean([b[1] for b in self.buf], axis=0)
        n = np.linalg.norm(quat)
        if n > 1e-9:
            quat /= n
        return pos, quat


# ------------------------------------------------------------
# Main node
# ------------------------------------------------------------
class QRPublisher:
    def __init__(self, args):
        rospy.init_node('qr_tf_publisher', anonymous=True)

        # TF
        self.tfb   = tf2_ros.TransformBroadcaster()
        self.buf   = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf)

        # RealSense
        self.pipe  = rs.pipeline()
        cfg        = rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height,
                          rs.format.bgr8, args.fps)
        self.pipe.start(cfg)

        # Intrinsics
        if args.intrinsics:
            self.K, self.dist = load_intrinsics(args.intrinsics)
            rospy.loginfo("Loaded intrinsics from %s", args.intrinsics)
        else:
            self.K = self.dist = None      # will auto‑fetch on first frame

        # ArUco
        self.dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, 'DICT_'+args.dictionary))
        self.par  = cv2.aruco.DetectorParameters()
        self.mlen = args.marker_length

        # Frames & filter
        self.base  = args.base_frame
        self.cam   = args.cam_frame
        self.qr    = args.qr_frame
        self.filt  = PoseFilter(args.filter_N)

        # Display
        self.gui = not args.no_display
        if self.gui:
            cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    # --------------------------------------------------------
    # Run loop
    # --------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            base2cam = self.lookup_tf(self.base, self.cam)
            frame    = self.pipe.wait_for_frames().get_color_frame()
            img      = cv2.cvtColor(np.asanyarray(frame.get_data()),
                                    cv2.COLOR_RGB2BGR)

            cam2qr = self.detect_marker(img)
            if base2cam is not None and cam2qr is not None:
                base2qr = base2cam @ cam2qr
                p = base2qr[:3, 3]
                q = tft.quaternion_from_matrix(base2qr)
                self.filt.add(p, q)
                out = self.filt.get()
                if out:
                    self.publish_tf(out[0], out[1])

            if self.gui:
                overlay_help(img, "q: quit")
                cv2.imshow("Live", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            rate.sleep()

        self.pipe.stop()
        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # TF lookup parent → child  → 4×4
    # --------------------------------------------------------
    def lookup_tf(self, parent, child):
        try:
            ts = self.buf.lookup_transform(parent, child, rospy.Time(0))
        except Exception:
            return None
        T = np.eye(4)
        t = ts.transform.translation
        T[:3, 3] = [t.x, t.y, t.z]
        q = ts.transform.rotation
        R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        T[:3, :3] = R
        return T

    # --------------------------------------------------------
    # Detect first ArUco marker → cam→qr 4×4
    # --------------------------------------------------------
    def detect_marker(self, img_bgr):
        if self.K is None:     # fetch intrinsics once
            intr = self.pipe.get_active_profile() \
                      .get_stream(rs.stream.color) \
                      .as_video_stream_profile() \
                      .get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float32)
            self.dist = np.array(intr.coeffs[:5], dtype=np.float32)
            rospy.loginfo("Intrinsics auto-fetched from RealSense")

        corners, ids, _ = cv2.aruco.detectMarkers(img_bgr, self.dict,
                                                  parameters=self.par)
        if ids is None:
            return None
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[0], self.mlen, self.K, self.dist)
        cv2.aruco.drawDetectedMarkers(img_bgr, [corners[0]])
        cv2.drawFrameAxes(img_bgr, self.K, self.dist,
                          rvec[0], tvec[0], self.mlen*0.5)
        R,_ = cv2.Rodrigues(rvec[0])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = tvec[0].flatten()
        return T

    # --------------------------------------------------------
    # Publish TF  base → qr  (dynamic broadcaster)
    # --------------------------------------------------------
    def publish_tf(self, pos, quat):
        ts = TransformStamped()
        ts.header.stamp = rospy.Time.now()
        ts.header.frame_id  = self.base
        ts.child_frame_id   = self.qr
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = pos
        ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w = quat
        self.tfb.sendTransform(ts)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Publish base→QR TF using RealSense & ArUco")
    ap.add_argument('--intrinsics',  default="camIntrinsic.yaml", help="YAML with camera_matrix & dist_coeff")
    ap.add_argument('--width',       type=int, default=1280)
    ap.add_argument('--height',      type=int, default=720)
    ap.add_argument('--fps',         type=int, default=30)
    ap.add_argument('--marker-length',type=float, default=0.04)
    ap.add_argument('--dictionary',
                    choices=['4X4_50','5X5_100','6X6_250','7X7_1000'],
                    default='6X6_250')
    ap.add_argument('--base-frame',  default='panda_hand')
    ap.add_argument('--cam-frame',   default='rs_camera')
    ap.add_argument('--qr-frame',    default='object_frame')
    ap.add_argument('--filter-N',    type=int, default=10,
                    help="window size for moving-average filter")
    ap.add_argument('--no-display',  action='store_true')
    args = ap.parse_args()

    QRPublisher(args).spin()


if __name__ == '__main__':
    main()
