#!/usr/bin/env python3
"""
Azure Kinect + cube-side ArUco markers -> /cube_tf (geometry_msgs/PoseStamped)

Same estimator as 6_publish_box_tf.py (joint PnP + Kalman gap fill, TF,
MarkerArray, fail-safe stop after --max-predict); only the object model and
defaults differ. 6_publish_box_tf.py must stay next to this file.

* Pose = CUBE CENTER, frame_id = --map-frame (default 'base', required by the
  lift policy; a cube standing on the desk should read z ~ 0.395).
* Cube: 50 (x) x 50 (y) x 100 (z) mm, two 50 mm cubes stacked; one 40 mm
  ArUco (DICT_6X6_250) per side face, centered in the upper or lower half
  (z = +-25 mm from the cube center).

Cube frame: z up, x = normal of the face carrying ID0 (the face turned toward
the camera, i.e. +x of base -> yaw 0 when ID0 faces the camera), y = z cross x.

Default layout (--layout ID:FACE:LEVEL[:ROT_DEG]):
    0:+x:low   1:+y:high   2:-x:low   3:-y:high
  (going around the cube to the right as seen from outside: ID0 lower ->
   ID1 upper -> ID2 lower -> ID3 upper)
  FACE  = outward normal of the side face (+x, -x, +y, -y)
  LEVEL = high (z=+25 mm) / low (z=-25 mm) / mid (z=0)
  ROT   = in-plane rotation of the printed marker, CCW seen from outside
          (default 0: marker "up" = cube +z, marker right = right as seen
          from outside the face)
Verify with --check (each visible marker's implied cube pose is logged; with
a correct layout adjacent markers agree within a few mm / deg).

Run (stop 6_publish_box_tf.py first: only one process can open the camera,
and keep the big box out of view - it uses the same IDs 0..3):
  python3 7_publish_cube_tf.py
  python3 7_publish_cube_tf.py --check          # verify layout
  python3 7_publish_cube_tf.py --width 1920 --height 1080   # small/far cube
"""

import importlib.util
import os
import sys

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("box_tf6", os.path.join(HERE, "6_publish_box_tf.py"))
box6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(box6)

FACE_NORMALS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}
DEFAULT_LAYOUT = ["0:+x:low", "1:+y:high", "2:-x:low", "3:-y:high"]


def parse_layout(entries):
    out = {}
    for s in entries:
        p = s.split(":")
        if len(p) not in (3, 4) or p[1] not in FACE_NORMALS or p[2] not in ("high", "low", "mid"):
            raise ValueError(f"bad --layout entry '{s}' (expected ID:+x|-x|+y|-y:high|low|mid[:ROT])")
        out[int(p[0])] = (p[1], p[2], float(p[3]) if len(p) == 4 else 0.0)
    return out


def build_cube_model(args):
    """{id: (4,3) corners in CUBE frame}, order TL, TR, BR, BL (ArUco)."""
    hx, hy, hz = args.box_x / 2.0, args.box_y / 2.0, args.box_z / 2.0
    zlev = {"high": hz / 2.0, "low": -hz / 2.0, "mid": 0.0}
    h = args.marker_length / 2.0
    model = {}
    for mid, (face, level, rot) in parse_layout(args.layout).items():
        n = np.array(FACE_NORMALS[face], np.float64)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, n)                    # right as seen from outside the face
        if rot:
            Rn = Rotation.from_rotvec(n * np.deg2rad(rot)).as_matrix()
            right, up = Rn @ right, Rn @ up
        c = n * np.array([hx, hy, hz]) + np.array([0.0, 0.0, zlev[level]])
        model[mid] = np.array([c - h * right + h * up, c + h * right + h * up,
                               c + h * right - h * up, c - h * right - h * up], np.float64)
    return model


class CubeTFPublisher(box6.BoxTFPublisher):
    def log_marker_yaws(self, corners, ids):
        """--check: cube pose implied by each visible marker on its own."""
        loc = box6.marker_local_corners(self.args.marker_length)
        poses = {}
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i not in self.model:
                continue
            ok, rv, tv = cv2.solvePnP(loc, c.reshape(4, 2).astype(np.float64),
                                      self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            m = self.model[i]
            r = (m[1] - m[0]) / np.linalg.norm(m[1] - m[0])
            u = (m[0] - m[3]) / np.linalg.norm(m[0] - m[3])
            R_cm = np.column_stack([r, u, np.cross(r, u)])     # marker axes in cube frame
            R = Rotation.from_rotvec(rv.ravel()) * Rotation.from_matrix(R_cm).inv()
            t = tv.ravel() - R.apply(m.mean(0))
            poses[i] = (R, t)
        if not poses:
            return
        ref = min(poses)
        R0, t0 = poses[ref]
        msg = []
        for i, (R, t) in sorted(poses.items()):
            zt = R.apply([0, 0, 1])
            tilt = np.degrees(np.arccos(np.clip(-zt[1], -1, 1)))  # cam y is down
            s = f"ID{i}: center cam ({t[0]*1e3:.0f},{t[1]*1e3:.0f},{t[2]*1e3:.0f})mm tilt {tilt:.0f}deg"
            if i != ref:
                d = (R0.inv() * R).magnitude()
                s += f" | vs ID{ref}: {np.linalg.norm(t - t0)*1e3:.0f}mm {np.degrees(d):.0f}deg"
            msg.append(s)
        self.get_logger().info(" || ".join(msg), throttle_duration_sec=1.0)


def build_arg_parser():
    ap = box6.build_arg_parser()
    ap.description = "Publish /cube_tf (cube center PoseStamped) from ArUco"
    ap.add_argument("--layout", nargs="*", default=DEFAULT_LAYOUT,
                    help="markers as ID:FACE:LEVEL[:ROT_DEG], see the file header")
    ap.set_defaults(
        topic="/cube_tf",
        box_frame="cube",
        map_frame="base_link",
        marker_length=0.04,
        box_x=0.05, box_y=0.05, box_z=0.10,
        extra_markers=os.path.join(HERE, "cube_extra_markers.yaml"),
    )
    return ap


def main():
    args = build_arg_parser().parse_args()
    box6.build_marker_model = build_cube_model      # used by BoxTFPublisher.setup_estimator

    rclpy.init(args=["--ros-args", "-r", "__node:=cube_tf_publisher"])
    node = CubeTFPublisher(args)
    try:
        node.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
