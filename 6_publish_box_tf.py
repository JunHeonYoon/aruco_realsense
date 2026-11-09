#!/usr/bin/env python3
"""
Azure Kinect + box-face ArUco markers -> /box_tf (geometry_msgs/PoseStamped)

* Pose = BOX CENTER (volume center) expressed in the camera frame.
* Markers: top face ID 0..4 (layout below) + any markers on other faces
  learned with --learn and stored in --extra-markers (YAML). Works with any
  subset visible, from any face. Per frame:
  1. Detect markers, then recover missed ones:
     - board reprojection + bit check (cv2.aruco.refineDetectedMarkers)
     - rejected quads matched to the filter prediction (motion blur: the black
       border is still found but the bits are unreadable). Only allowed for
       --recover-window after the last frame with a decoded marker.
  2. One joint PnP over all corners: IPPE init on the face with the most
     visible markers (candidates scored on all points, planar ambiguity
     resolved with the prediction), then Huber-IRLS Gauss-Newton refinement.
     A marker whose residual disagrees with the others is dropped.
  3. Measurement covariance from the PnP Jacobian, sigma^2 (J^T W J)^-1:
     one small/far marker -> large uncertainty, many markers -> small.
* 12-state error-state Kalman filter (constant velocity for position,
  constant angular velocity for orientation) weights each measurement by its
  own covariance, chi-square gates outliers and fills short dropouts with
  predictions.
* Also broadcasts TF <cam-frame> -> <box-frame> (default 'box'), same pose
  and stamp as /box_tf (disable with --no-tf).
* With --map-frame NAME, /box_tf and the marker are expressed in NAME instead
  of the camera frame (NAME -> <cam-frame> looked up from TF, e.g. the
  base_link -> camera static TF of 4_publish_basecam_tf.py).
* /box_tf/marker (visualization_msgs/MarkerArray): the box as a CUBE, green =
  measured, orange = predicted. Deleted when tracking is lost (no lifetime,
  so RViz on another machine with a different clock still shows it).
* If no marker has been accepted for longer than --max-predict (default 0.2 s),
  nothing is published (robot side faults after 0.3 s -> fail safe).

Box frame : x -> toward camera, y -> robot's left (ID0/ID3 side), z -> up
Box size  : 410 (y, left-right) x 310 (x, front-back) x 280 (z) mm

Top-face marker layout (seen from the camera, robot on the far side):

        robot side (-x)
     ID1 +-----------+ ID0
         |    ID4    |          <- +y is to the RIGHT in this picture
     ID2 +-----------+ ID3
        camera side (+x)

Marker centers of ID0..3 are --edge-y (4.5 cm) from the left/right edge
and --edge-x (5 cm) from the front/back edge. ID4 is at the face center.

Assumed marker orientation: every marker printed the same way, with the
marker's "up" (top edge in the printed image) pointing to the ROBOT side
(-x) and the marker's right pointing to +y. If a marker is mounted rotated,
fix it with --marker-yaw ID:DEG (run with --check to see per-marker yaw).

Markers on other faces (e.g. the front face) -- learn them once:
  python3 6_publish_box_tf.py --learn --learn-ids 5 6 7 8 9
Hold the box so the top face (>= 2 markers) and the other face are both
visible, move it slowly through a few poses. Each new marker's placement in
the box frame is optimized over --learn-samples frames, saved to
--extra-markers and used immediately; later runs load it automatically.

Keys (display window): q quit
"""

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

from azure_kinect_camera import add_exposure_args, get_camera_intrinsics, start_azure_kinect


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def build_marker_model(args):
    """Return {id: (4,3) corner points in BOX frame} (TL, TR, BR, BL order)."""
    L_y, L_x, H = args.box_y, args.box_x, args.box_z
    cx = L_x / 2.0 - args.edge_x   # |x| of corner-marker centers
    cy = L_y / 2.0 - args.edge_y   # |y| of corner-marker centers
    zt = H / 2.0                   # top face height from box center

    centers = {
        0: (-cx, +cy),
        1: (-cx, -cy),
        2: (+cx, -cy),
        3: (+cx, +cy),
        4: (0.0, 0.0),
    }

    yaw_override = {}
    for s in args.marker_yaw:
        mid, deg = s.split(":")
        yaw_override[int(mid)] = float(deg)

    h = args.marker_length / 2.0
    # marker-local corners: x = marker right, y = marker up (ArUco order TL,TR,BR,BL)
    local = np.array([[-h, +h], [+h, +h], [+h, -h], [-h, -h]])
    # marker right -> box +y, marker up -> box -x
    base_R = np.array([[0.0, -1.0],
                       [1.0, 0.0]])  # columns: marker x in box xy, marker y in box xy

    model = {}
    for mid, (x0, y0) in centers.items():
        yaw = np.deg2rad(yaw_override.get(mid, 0.0))  # rotation about box +z
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        xy = (Rz @ base_R @ local.T).T + np.array([x0, y0])
        model[mid] = np.hstack([xy, np.full((4, 1), zt)]).astype(np.float64)
    return model


def marker_local_corners(marker_length):
    """Corners in the marker's own frame (x right, y up, z out), TL, TR, BR, BL."""
    h = marker_length / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def marker_normal(c):
    """Outward normal (marker +z) of a (4,3) corner set in TL, TR, BR, BL order."""
    n = np.cross(c[3] - c[0], c[1] - c[0])
    return n / np.linalg.norm(n)


def load_extra_markers(path):
    """{id: (4,3) corners in BOX frame} from a YAML written by --learn."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {int(k): np.array(v, np.float64).reshape(4, 3)
            for k, v in (data.get("markers") or {}).items()}


def save_extra_markers(path, markers):
    data = {
        "description": "ArUco marker corners in the BOX frame [m], order TL, TR, BR, BL "
                       "(written by 6_publish_box_tf.py --learn)",
        "markers": {int(k): np.round(v, 5).tolist() for k, v in sorted(markers.items())},
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


# ----------------------------------------------------------------------------
# Robust PnP refinement + covariance
# ----------------------------------------------------------------------------
def project(obj, R, t, K, dist):
    proj, _ = cv2.projectPoints(obj, R.as_rotvec(), np.asarray(t, np.float64), K, dist)
    return proj.reshape(-1, 2)


def numeric_jacobian(f, R, t, eps=1e-6):
    """Residuals f(R, t) and their Jacobian w.r.t. [d_t, d_theta(body)]."""
    r0 = f(R, t)
    J = np.empty((r0.size, 6))
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        J[:, k] = (f(R, t + d) - r0) / eps
        J[:, 3 + k] = (f(R * Rotation.from_rotvec(d), t) - r0) / eps
    return r0, J


def huber_weights(e, k):
    w = np.ones_like(e)
    big = e > k
    w[big] = k / e[big]
    return w


def depth_residuals(R, t, dz):
    """
    Measured minus model depth [m] of depth samples. Sample i lies on the ray
    dz["rays"][i] (z = 1); the model depth is where that ray meets its box face
    (normal dz["n"][i] through point dz["o"][i], box frame) under pose (R, t).
    """
    n_c = R.apply(dz["n"])
    o_c = R.apply(dz["o"]) + t
    z_model = np.sum(n_c * o_c, axis=1) / np.sum(n_c * dz["rays"], axis=1)
    return dz["z"] - z_model


def refine_pose(obj, imgp, w0, R, t, K, dist, huber_px, pix_sigma, dz=None, iters=10):
    """
    Huber-IRLS Gauss-Newton on X_cam = R exp(d_theta) X_box + (t + d_t).
    Residuals are normalized by their noise: corners / pix_sigma and, with
    dz (depth samples), depth / dz["sigma"] (Huber at 3 sigma).
    w0: per-corner prior weights (1 / sigma_scale^2).
    Returns R, t, per-corner residual norms [px], 6x6 covariance of [t, theta]
    (sigma^2 (J^T W J)^-1, scaled up if the residuals are larger than the noise).
    """
    n2 = 2 * len(imgp)

    def resid(R, t):
        r = (project(obj, R, t, K, dist) - imgp).ravel() / pix_sigma
        if dz is not None:
            r = np.concatenate([r, depth_residuals(R, t, dz) / dz["sigma"]])
        return r

    def weights(r):
        e = np.linalg.norm(r[:n2].reshape(-1, 2), axis=1)
        W = np.repeat(w0 * huber_weights(e, huber_px / pix_sigma), 2)
        if dz is not None:
            W = np.concatenate([W, huber_weights(np.abs(r[n2:]), 3.0)])
        return W

    for _ in range(iters):
        r, J = numeric_jacobian(resid, R, t)
        W = weights(r)
        delta = -np.linalg.solve(J.T @ (W[:, None] * J) + 1e-12 * np.eye(6), J.T @ (W * r))
        t = t + delta[:3]
        R = R * Rotation.from_rotvec(delta[3:])
        if np.linalg.norm(delta) < 1e-8:
            break
    r, J = numeric_jacobian(resid, R, t)
    W = weights(r)
    s2 = max(1.0, float(np.sum(W * r ** 2)) / max(r.size - 6, 1))
    cov = s2 * np.linalg.pinv(J.T @ (W[:, None] * J))
    e = np.linalg.norm(r[:n2].reshape(-1, 2), axis=1) * pix_sigma
    return R, t, e, cov


# ----------------------------------------------------------------------------
# Kalman filter (error-state, constant velocity / angular velocity)
# ----------------------------------------------------------------------------
class PoseKalman:
    """
    State [p, v, theta, omega] (12). theta is the body-frame error of the
    nominal rotation R and is folded into R after every step, so it stays 0.
    Measurement: [p, theta] with a full 6x6 covariance per measurement.
    """

    IDX = np.r_[0:3, 6:9]

    def __init__(self, q_pos, q_rot):
        self.q_pos = q_pos    # white-noise acceleration spectral density
        self.q_rot = q_rot    # white-noise angular acceleration spectral density
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.R = None         # nominal rotation (scipy Rotation)
        self.P = np.eye(12)
        self.t = None         # time of current state [s]

    @property
    def initialized(self):
        return self.R is not None

    def reset(self, t, p, R, cov):
        self.t = t
        self.p, self.v, self.w, self.R = p.copy(), np.zeros(3), np.zeros(3), R
        self.P = np.eye(12)
        self.P[np.ix_(self.IDX, self.IDX)] = cov

    def predict_to(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(12)
        F[0:3, 3:6] = dt * np.eye(3)
        F[6:9, 9:12] = dt * np.eye(3)
        Q = np.zeros((12, 12))
        blk = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        for q, (i, j) in ((self.q_pos, (0, 3)), (self.q_rot, (6, 9))):
            for a, ia in enumerate((i, j)):
                for b, ib in enumerate((i, j)):
                    Q[ia:ia + 3, ib:ib + 3] = q * blk[a, b] * np.eye(3)
        self.p = self.p + self.v * dt
        self.R = self.R * Rotation.from_rotvec(self.w * dt)
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _innovation(self, p, R, cov):
        y = np.concatenate([p - self.p, (self.R.inv() * R).as_rotvec()])
        S = self.P[np.ix_(self.IDX, self.IDX)] + cov
        return y, S

    def gate(self, p, R, cov):
        """Squared Mahalanobis distance (chi-square, 6 DOF) of a measurement."""
        y, S = self._innovation(p, R, cov)
        return float(y @ np.linalg.solve(S, y))

    def update(self, p, R, cov):
        y, S = self._innovation(p, R, cov)
        K = self.P[:, self.IDX] @ np.linalg.inv(S)          # (12,6)
        dx = K @ y
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = self.R * Rotation.from_rotvec(dx[6:9])
        self.w = self.w + dx[9:12]
        H = np.zeros((6, 12))
        H[np.arange(6), self.IDX] = 1.0
        I_KH = np.eye(12) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ cov @ K.T     # Joseph form

    def state(self):
        return self.p.copy(), self.R

    def pos_std(self):
        return np.sqrt(np.diag(self.P)[0:3])


# ----------------------------------------------------------------------------
# Learning markers on other faces
# ----------------------------------------------------------------------------
class MarkerLearner:
    """
    Estimate the placement (6-DOF, box frame) of markers that are not in the
    model yet, from frames where the box pose is well measured by known markers.
    Init: medoid of per-frame single-marker IPPE poses. Final: Huber Gauss-Newton
    over all frames (box poses fixed) minimizing the corner reprojection error.
    """

    def __init__(self, ids, n_samples, marker_length, K, dist):
        self.ids = set(ids)            # empty -> learn any unknown id
        self.n_samples = n_samples
        self.loc = marker_local_corners(marker_length)
        self.K, self.dist = K, dist
        self.samples = {}              # id -> [(R_box, t_box, corners_img)]

    def add(self, R, t, unknown):
        for mid, c in unknown.items():
            if self.ids and mid not in self.ids:
                continue
            side = float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))
            if side >= 20.0:
                self.samples.setdefault(mid, []).append((R, t, c))

    def ready(self):
        return [m for m, s in self.samples.items() if len(s) >= self.n_samples]

    def progress(self):
        return " ".join(f"{m}:{len(s)}/{self.n_samples}" for m, s in sorted(self.samples.items()))

    def _residuals(self, S, Rb, tb):
        obj = Rb.apply(self.loc) + tb
        return np.concatenate([(project(obj, R, t, self.K, self.dist) - c).ravel() for R, t, c in S])

    def solve(self, mid):
        """Returns ((4,3) corners in box frame, RMS [px]) and forgets the samples."""
        S = self.samples.pop(mid)
        inits = []
        for R, t, c in S:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self.loc, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n == 0:
                continue
            k = int(np.argmin(np.ravel(errs)))
            Rm = Rotation.from_rotvec(rvecs[k].ravel())
            inits.append((R.inv() * Rm, R.inv().apply(tvecs[k].ravel() - t)))
        sub = inits[::max(1, len(inits) // 30)]
        Rb, tb = min(sub, key=lambda a: sum((a[0].inv() * b[0]).magnitude()
                                              + 10.0 * np.linalg.norm(a[1] - b[1]) for b in inits))
        eps = 1e-6
        for _ in range(20):
            r = self._residuals(S, Rb, tb)
            J = np.empty((r.size, 6))
            for k in range(3):
                d = np.zeros(3)
                d[k] = eps
                J[:, k] = (self._residuals(S, Rb, tb + d) - r) / eps
                J[:, 3 + k] = (self._residuals(S, Rb * Rotation.from_rotvec(d), tb) - r) / eps
            e = np.linalg.norm(r.reshape(-1, 2), axis=1)
            W = np.repeat(huber_weights(e, 2.0), 2)
            delta = -np.linalg.solve(J.T @ (W[:, None] * J) + 1e-12 * np.eye(6), J.T @ (W * r))
            tb = tb + delta[:3]
            Rb = Rb * Rotation.from_rotvec(delta[3:])
            if np.linalg.norm(delta) < 1e-9:
                break
        r = self._residuals(S, Rb, tb)
        return Rb.apply(self.loc) + tb, float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2))


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------
class BoxTFPublisher(Node):
    def __init__(self, args):
        super().__init__("box_tf_publisher")
        self.args = args

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(PoseStamped, args.topic, qos)
        self.pub_pred = self.create_publisher(Bool, args.topic + "/predicted", qos)
        self.tfb = None if args.no_tf else tf2_ros.TransformBroadcaster(self)
        self.pub_marker = (None if args.no_marker else
                           self.create_publisher(MarkerArray, args.topic + "/marker", qos))
        self.marker_shown = False
        if args.map_frame:
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

        self.camera = start_azure_kinect(0, args.width, args.height, args.fps,
                                         exposure=args.exposure, gain=args.gain,
                                         powerline_hz=args.powerline,
                                         depth_mode=None if args.depth_mode == "OFF" else args.depth_mode)
        self.K, self.dist = get_camera_intrinsics(self.camera)
        self.K = self.K.astype(np.float64)
        self.dist = self.dist.astype(np.float64)
        self.setup_estimator(args)

        self.gui = not args.no_display
        if self.gui:
            cv2.namedWindow("box_tf", cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "marker centers (box frame, mm): "
            + ", ".join(f"ID{i}=({c.mean(0)[0]*1e3:.0f},{c.mean(0)[1]*1e3:.0f},{c.mean(0)[2]*1e3:.0f})"
                        for i, c in self.model.items())
        )

    def setup_estimator(self, args):
        """Detector, marker model, board and filter (no camera / ROS needed)."""
        self.dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_" + args.dictionary))
        self.par = cv2.aruco.DetectorParameters_create()
        self.par.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_" + args.corner_refine.upper())

        # box faces for depth sampling: (normal, center, u, v, half_u, half_v), box frame
        half = np.array([args.box_x, args.box_y, args.box_z]) / 2.0
        self.box_faces = []
        for k in range(3):
            i, j = [a for a in range(3) if a != k]
            for sgn in (1.0, -1.0):
                n = np.zeros(3)
                n[k] = sgn
                self.box_faces.append((n, n * half[k], np.eye(3)[i], np.eye(3)[j], half[i], half[j]))

        self.model = build_marker_model(args)
        self.extra = load_extra_markers(args.extra_markers)
        clash = sorted(set(self.extra) & set(self.model))
        if clash:
            raise ValueError(f"{args.extra_markers}: ids {clash} clash with the top-face ids")
        self.model.update(self.extra)
        self.update_model()
        self.learner = (MarkerLearner(args.learn_ids, args.learn_samples, args.marker_length,
                                      self.K, self.dist) if args.learn else None)
        # model error (marker placement, print scale) floor on measurement noise
        self.cov_floor = np.diag([args.r_pos**2] * 3 + [np.deg2rad(args.r_rot)**2] * 3)

        self.kf = PoseKalman(args.q_pos, args.q_rot)
        self.last_meas_t = None
        self.last_decoded_t = None
        self.depth_bias = None          # slowly varying depth - RGB offset [m]
        self.depth_bias_t = None
        self.n_reject = 0
        self.stats = dict(frames=0, no_det=0, rms_rej=0, gate_rej=0, soft=0, ok=0, rec=0, drop=0,
                          pub=0, pred=0)
        self.stats_t0 = time.monotonic()
        self.per_marker = {}
        self.last_info = ""
        self.gui = False

    def update_model(self):
        """Recompute per-marker normals and the refine board after the model changed."""
        self.normals = {m: marker_normal(c) for m, c in self.model.items()}
        mids = sorted(self.model)
        self.board = cv2.aruco.Board_create(
            [self.model[i].astype(np.float32) for i in mids], self.dict,
            np.array(mids, np.int32).reshape(-1, 1))

    def learn_step(self, meas):
        """Collect samples while the box pose is well measured; add finished markers."""
        a = self.args
        if len(meas["ids"]) >= 2 and meas["rms"] < 1.5 and meas["unknown"]:
            self.learner.add(meas["R"], meas["p"], meas["unknown"])
        for mid in self.learner.ready():
            corners, rms = self.learner.solve(mid)
            if rms > a.max_rms:
                self.get_logger().warn(f"learn ID{mid}: rms {rms:.2f}px too large, discarded")
                continue
            self.model[mid] = corners
            self.extra[mid] = corners
            self.update_model()
            save_extra_markers(a.extra_markers, self.extra)
            c = corners.mean(0) * 1e3
            n = self.normals[mid]
            self.get_logger().info(
                f"learned ID{mid}: center ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) mm, normal "
                f"({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}), rms {rms:.2f}px -> saved {a.extra_markers}")

    # -- capture --------------------------------------------------------------
    def grab(self):
        cap = self.camera.get_capture()
        color = cap.color
        if color is None:
            return None, None
        sys_ns = cap.color_system_timestamp_nsec   # perf_counter_ns() clock
        stamp_ns = time.time_ns() - (time.perf_counter_ns() - sys_ns)
        img = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR) if color.shape[2] == 4 else color.copy()
        self.depth = cap.transformed_depth if self.args.depth_mode != "OFF" else None
        return img, stamp_ns

    # -- measurement ----------------------------------------------------------
    def measure(self, img, prediction, allow_recover, depth=None, t_now=None):
        """
        prediction: (p, R) from the filter, or None if it is stale.
        depth: depth image registered to the color image [mm], or None.
        Returns dict(p, R, cov, rms, ids, rec, dropped, unknown, n_depth) or None.
        """
        a = self.args
        corners, ids, rejected = cv2.aruco.detectMarkers(img, self.dict, parameters=self.par)
        if ids is not None and len(ids) > 0:
            try:
                corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                    img, self.board, corners, ids, rejected,
                    cameraMatrix=self.K, distCoeffs=self.dist, parameters=self.par)
            except cv2.error:
                pass        # board pose not solvable from these markers; keep detections

        found, unknown = {}, {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                (found if int(i) in self.model else unknown)[int(i)] = c.reshape(4, 2).astype(np.float64)
            if self.gui:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
            if a.check:
                self.log_marker_yaws(corners, ids)

        rec = {}
        if allow_recover and prediction is not None and len(found) < len(self.model):
            rec = self.recover_markers(rejected, found, prediction)
        if not found and not rec:
            return None

        used = sorted(found) + sorted(rec)
        obj = np.concatenate([self.model[m] for m in used])
        imgp = np.concatenate([found[m] if m in found else rec[m] for m in used])
        w0 = np.concatenate([np.full(4, 1.0 if m in found else 1.0 / a.recover_sigma**2)
                             for m in used])

        init = self.solve_init(used, obj, imgp, prediction)
        if init is None:
            return None
        R, t, inl = init
        dropped = [m for m, ok in zip(used, inl) if not ok]
        if dropped:
            keep = np.repeat(inl, 4)
            obj, imgp, w0 = obj[keep], imgp[keep], w0[keep]
            used = [m for m, ok in zip(used, inl) if ok]
            for m in dropped:
                found.pop(m, None)
                rec.pop(m, None)

        # depth samples on the visible faces, taken around the RGB-only pose.
        # Only while the box is nearly still: the ToF image is not exposed at the
        # same instant as the color image, so it disagrees with RGB during motion.
        dz = None
        still = (prediction is None
                 or (np.linalg.norm(self.kf.v) < a.depth_max_speed
                     and np.linalg.norm(self.kf.w) < a.depth_max_rot_speed))
        if depth is not None and still:
            R, t, _, _ = refine_pose(obj, imgp, w0, R, t, self.K, self.dist, a.huber_px, a.pix_sigma)
            dz = self.sample_depth(depth, R, t)
            if dz is not None and a.depth_bias_tau > 0:
                dz["z"] = dz["z"] - self.update_depth_bias(dz, R, t, t_now)

        # robust refinement; drop a marker that disagrees with the rest, re-solve
        def solve(dz):
            R1, t1, o1, i1, w1, u1, drop1 = R, t, obj, imgp, w0, list(used), []
            while True:
                R1, t1, e1, cov1 = refine_pose(o1, i1, w1, R1, t1, self.K, self.dist, a.huber_px,
                                               a.pix_sigma, dz)
                per1 = {m: float(np.sqrt(np.mean(e1[4 * k:4 * k + 4] ** 2))) for k, m in enumerate(u1)}
                if len(u1) < 2:
                    break
                worst = max(per1, key=per1.get)
                others = np.median([v for m, v in per1.items() if m != worst])
                if per1[worst] <= max(a.outlier_px, 3.0 * others):
                    break
                k = u1.index(worst)
                keep = np.r_[0:4 * k, 4 * k + 4:len(i1)]
                o1, i1, w1 = o1[keep], i1[keep], w1[keep]
                u1.pop(k)
                drop1.append(worst)
            return R1, t1, e1, cov1, per1, o1, drop1

        R, t, e, cov, per, obj, drop = solve(dz)
        if dz is not None and float(np.sqrt(np.mean(e ** 2))) > a.max_rms:
            dz = None                   # depth disagrees with the markers -> RGB only
            R, t, e, cov, per, obj, drop = solve(None)
        dropped += drop
        for m in drop:
            found.pop(m, None)
            rec.pop(m, None)

        if np.any(R.apply(obj)[:, 2] + t[2] <= 0):
            return None                 # converged to a mirror solution behind the camera
        cov = cov + self.cov_floor
        rms = float(np.sqrt(np.mean(e ** 2)))
        self.per_marker = per

        if self.gui:
            for c in rec.values():
                cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (0, 165, 255), 2)
            cv2.drawFrameAxes(img, self.K, self.dist, R.as_rotvec(), t, 0.1)
        return dict(p=t, R=R, cov=cov, rms=rms, ids=sorted(found), rec=sorted(rec), dropped=dropped,
                    unknown=unknown, n_depth=0 if dz is None else len(dz["z"]))

    def update_depth_bias(self, dz, R, t, t_now):
        """
        Track the constant depth-vs-RGB offset (ToF bias, or marker layout scale)
        with a time constant of --depth-bias-tau, so the absolute distance follows
        the long-term RGB average while depth still steadies every frame.
        """
        b = float(np.median(depth_residuals(R, t, dz)))
        if self.depth_bias is None or t_now is None:
            self.depth_bias, self.depth_bias_t = b, t_now
        elif abs(b - self.depth_bias) < self.args.depth_gate:
            alpha = min(1.0, max(t_now - self.depth_bias_t, 0.0) / self.args.depth_bias_tau)
            self.depth_bias += alpha * (b - self.depth_bias)
            self.depth_bias_t = t_now
        return self.depth_bias

    def sample_depth(self, depth, R, t):
        """
        Depth samples on a grid over each box face that faces the camera
        (margin from the edges, no grazing views), kept only where the depth
        agrees with the RGB pose within --depth-gate (drops hands, occluders).
        """
        a = self.args
        h, w = depth.shape[:2]
        cam_box = R.inv().apply(-t)          # camera position in box frame
        g = np.linspace(-a.depth_margin, a.depth_margin, a.depth_grid)
        gu, gv = [x.reshape(-1, 1) for x in np.meshgrid(g, g)]
        rays, zs, ns, os_ = [], [], [], []
        for n, o, u, v, hu, hv in self.box_faces:
            view = cam_box - o
            if n @ view / np.linalg.norm(view) < a.depth_min_cos:
                continue
            P = o + gu * hu * u + gv * hv * v
            pi = np.round(project(P, R, t, self.K, self.dist)).astype(int)
            ok = (pi[:, 0] >= 0) & (pi[:, 0] < w) & (pi[:, 1] >= 0) & (pi[:, 1] < h)
            if not ok.any():
                continue
            pi = pi[ok]
            z = depth[pi[:, 1], pi[:, 0]].astype(np.float64) * 1e-3
            xy = cv2.undistortPoints(pi.astype(np.float64).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
            ray = np.column_stack([xy, np.ones(len(xy))])
            d = {"rays": ray, "z": z, "n": np.tile(n, (len(z), 1)), "o": np.tile(o, (len(z), 1))}
            good = (z > 0) & (np.abs(depth_residuals(R, t, d)) < a.depth_gate)
            rays.append(ray[good])
            zs.append(z[good])
            ns.append(d["n"][good])
            os_.append(d["o"][good])
        if not zs or sum(len(z) for z in zs) < 10:
            return None
        return {"rays": np.concatenate(rays), "z": np.concatenate(zs), "n": np.concatenate(ns),
                "o": np.concatenate(os_), "sigma": a.depth_sigma}

    def ippe_face(self, n, obj, imgp, ks):
        """Both IPPE solutions (R, t) of box pose from the markers ks lying on the face with normal n."""
        idx = np.concatenate([np.arange(4 * k, 4 * k + 4) for k in ks])
        # OpenCV 4.5.4 IPPE needs z=0 points: solve in a face-plane frame
        # (columns u, v, n; origin o on the face), then map back to the box frame.
        pts = obj[idx]
        o = pts.mean(0)
        u = pts[1] - pts[0]
        u = u - (u @ n) * n
        u /= np.linalg.norm(u)
        Rf = np.column_stack([u, np.cross(n, u), n])
        q = (pts - o) @ Rf
        q[:, 2] = 0.0
        cnt, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            q, imgp[idx], self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        sols = []
        for k in range(cnt):
            R = Rotation.from_matrix(Rotation.from_rotvec(rvecs[k].ravel()).as_matrix() @ Rf.T)
            t = tvecs[k].ravel() - R.apply(o)
            # the face must face the camera: its normal points toward the camera origin
            if R.apply(n) @ -(R.apply(o) + t) > 0:
                sols.append((R, t))
        return sols

    def solve_init(self, used, obj, imgp, prediction):
        """
        PnP init, robust to a wrong marker (e.g. the same id elsewhere in the
        scene). Hypotheses: IPPE on each face's markers and on every
        leave-one-marker-out subset of them. A hypothesis must put the whole
        model in front of the camera. Best = most markers agreeing (per-marker
        RMS < --init-px), then lowest median RMS; the planar ambiguity is
        resolved with the prediction.
        Returns (R, t, inlier mask over used) or None.
        """
        faces = []                      # [(normal, [index into used])]
        for k, m in enumerate(used):
            n = self.normals[m]
            for f in faces:
                if f[0] @ n > 0.9:
                    f[1].append(k)
                    break
            else:
                faces.append((n, [k]))
        subsets = []
        for n, ks in faces:
            subsets.append((n, ks))
            if len(ks) >= 2:
                subsets += [(n, [k for k in ks if k != drop]) for drop in ks]

        hyps = []
        for n, ks in subsets:
            for R, t in self.ippe_face(n, obj, imgp, ks):
                if np.any(R.apply(obj)[:, 2] + t[2] <= 0):
                    continue            # mirror solution behind the camera
                e2 = np.sum((project(obj, R, t, self.K, self.dist) - imgp) ** 2, axis=1)
                per = np.sqrt(e2.reshape(-1, 4).mean(1))
                inl = per < self.args.init_px
                if inl.any():
                    hyps.append((int(inl.sum()), float(np.median(per[inl])), R, t, inl))
        if not hyps:
            return None
        n_best = max(h[0] for h in hyps)
        top = sorted((h for h in hyps if h[0] == n_best), key=lambda h: h[1])
        best = top[0]
        if prediction is not None:
            # planar ambiguity: similar fits -> the one closest to the prediction
            close = [h for h in top if h[1] < 1.5 * best[1] + 0.5]
            Rp = prediction[1]
            best = min(close, key=lambda h: (Rp.inv() * h[2]).magnitude())
        return best[2], best[3], best[4]

    def recover_markers(self, rejected, found, prediction):
        """
        Match rejected quads (border found, bits unreadable, e.g. motion blur)
        to missing markers projected with the predicted pose.
        """
        cands = [np.asarray(c, np.float64).reshape(4, 2) for c in rejected]
        if not cands:
            return {}
        p_pred, R_pred = prediction
        taken, rec = set(), {}
        for mid, obj in self.model.items():
            if mid in found:
                continue
            c0 = R_pred.apply(obj.mean(0)) + p_pred
            if R_pred.apply(self.normals[mid]) @ -c0 <= 0:
                continue    # marker's face points away from the camera
            proj = project(obj, R_pred, p_pred, self.K, self.dist)
            side = float(np.mean(np.linalg.norm(proj - np.roll(proj, -1, axis=0), axis=1)))
            if side < 10.0:
                continue
            scores = []
            for j, c in enumerate(cands):
                if j in taken:
                    continue
                d, s = min((float(np.mean(np.linalg.norm(np.roll(c, s, axis=0) - proj, axis=1))), s)
                           for s in range(4))
                scores.append((d, s, j))
            if not scores:
                continue
            scores.sort()
            d, s, j = scores[0]
            if d > self.args.recover_frac * side:
                continue
            if len(scores) > 1 and scores[1][0] < 2.0 * d:
                continue    # two quads fit about equally well -> ambiguous
            c = np.roll(cands[j], s, axis=0)
            area = cv2.contourArea(c.astype(np.float32)) / max(cv2.contourArea(proj.astype(np.float32)), 1.0)
            if not 0.5 < area < 2.0:
                continue
            taken.add(j)
            rec[mid] = c
        return rec

    def log_marker_yaws(self, corners, ids):
        """Print each marker's yaw relative to its expected orientation."""
        h = self.args.marker_length / 2.0
        loc = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        Rs = {}
        for c, i in zip(corners, ids.flatten()):
            ok, rv, tv = cv2.solvePnP(loc, c.reshape(4, 2).astype(np.float64),
                                      self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                Rs[int(i)] = Rotation.from_rotvec(rv.ravel())
        if 4 in Rs:
            msg = []
            for i, R in sorted(Rs.items()):
                rel = (Rs[4].inv() * R).as_euler("xyz", degrees=True)
                msg.append(f"ID{i}: yaw vs ID4 = {rel[2]:+.1f} deg")
            self.get_logger().info(" | ".join(msg), throttle_duration_sec=1.0)

    # -- estimation step ------------------------------------------------------
    def step(self, img, t_now, depth=None):
        """Measure + filter one frame. Returns (meas, predicted, publish)."""
        a = self.args
        st = self.stats
        st["frames"] += 1
        self.per_marker = {}

        fresh = (self.kf.initialized and self.last_meas_t is not None
                 and t_now - self.last_meas_t <= a.reinit_after)
        if fresh:
            self.kf.predict_to(t_now)
        prediction = self.kf.state() if fresh else None
        allow_recover = (not a.no_recover and self.last_decoded_t is not None
                         and t_now - self.last_decoded_t <= a.recover_window)

        meas = self.measure(img, prediction, allow_recover, depth, t_now)
        if meas is None:
            st["no_det"] += 1
        elif meas["rms"] > a.max_rms:
            st["rms_rej"] += 1
            meas = None

        predicted = True
        if meas is not None:
            st["rec"] += int(bool(meas["rec"]))
            st["drop"] += len(meas["dropped"])
            accept = True
            d2 = self.kf.gate(meas["p"], meas["R"], meas["cov"]) if fresh else 0.0
            if not fresh:
                self.kf.reset(t_now, meas["p"], meas["R"], meas["cov"])
            elif d2 > a.gate:
                self.n_reject += 1
                if self.n_reject > a.max_reject:
                    # consistently far from the prediction: the box really moved
                    self.kf.reset(t_now, meas["p"], meas["R"], meas["cov"])
                else:
                    st["gate_rej"] += 1      # outlier -> keep prediction
                    accept = False
            else:
                # robust update: an unlikely (but not rejected) measurement gets its
                # covariance inflated, so a single-frame spike (mostly depth along
                # the camera ray) only nudges the state instead of pulling it along
                scale = max(1.0, d2 / a.soft_gate)
                st["soft"] += int(scale > 1.0)
                self.kf.update(meas["p"], meas["R"], meas["cov"] * scale)
            if accept:
                self.last_meas_t = t_now
                if meas["ids"]:
                    self.last_decoded_t = t_now
                self.n_reject = 0
                predicted = False
                st["ok"] += 1
                if self.learner is not None:
                    self.learn_step(meas)

        publish = (self.kf.initialized and self.last_meas_t is not None
                   and t_now - self.last_meas_t <= a.max_predict)
        if publish:
            st["pub"] += 1
            st["pred"] += int(predicted)
        if self.per_marker:
            self.last_info = "ids/rms(px): " + " ".join(
                f"{i}:{e:.1f}" for i, e in sorted(self.per_marker.items()))
        return meas, predicted, publish

    # -- main loop ------------------------------------------------------------
    def spin(self):
        a = self.args
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                img, stamp_ns = self.grab()
                if img is None:
                    continue
                t_now = stamp_ns * 1e-9

                meas, predicted, publish = self.step(img, t_now, self.depth)
                self.log_stats()
                if publish:
                    pos, Rk = self.kf.state()
                    if a.raw and not predicted:
                        pos, Rk = meas["p"], meas["R"]
                    self.publish(stamp_ns, pos, Rk, predicted)
                else:
                    self.delete_marker()

                if self.gui:
                    self.draw_status(img, meas, predicted, publish, t_now)
                    cv2.imshow("box_tf", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

    def log_stats(self):
        now = time.monotonic()
        dt = now - self.stats_t0
        if dt < 1.0:
            return
        st = self.stats
        self.get_logger().info(
            f"fps={st['frames']/dt:.1f} | meas_ok={st['ok']} no_det={st['no_det']} "
            f"rms_rej={st['rms_rej']} gate_rej={st['gate_rej']} soft={st['soft']} recovered={st['rec']} "
            f"dropped={st['drop']} | published={st['pub']} (pred {st['pred']}) | "
            f"last {self.last_info}")
        for k in st:
            st[k] = 0
        self.stats_t0 = now

    def cam_to_map(self):
        """Latest <map-frame> -> <cam-frame> as (R, t), or None if not available."""
        try:
            tf = self.tf_buf.lookup_transform(self.args.map_frame, self.args.cam_frame, Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f"no TF {self.args.map_frame} -> {self.args.cam_frame}: {e}",
                throttle_duration_sec=2.0)
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        return Rotation.from_quat([q.x, q.y, q.z, q.w]), np.array([tr.x, tr.y, tr.z])

    def publish(self, stamp_ns, pos, R, predicted):
        a = self.args
        header = PoseStamped().header
        header.stamp.sec = int(stamp_ns // 1_000_000_000)
        header.stamp.nanosec = int(stamp_ns % 1_000_000_000)

        if self.tfb is not None:
            ts = TransformStamped()
            ts.header = header
            ts.header.frame_id = a.cam_frame
            ts.child_frame_id = a.box_frame
            tr, q = ts.transform.translation, R.as_quat()
            tr.x, tr.y, tr.z = map(float, pos)
            ts.transform.rotation.x, ts.transform.rotation.y = float(q[0]), float(q[1])
            ts.transform.rotation.z, ts.transform.rotation.w = float(q[2]), float(q[3])
            self.tfb.sendTransform(ts)

        frame = a.cam_frame
        if a.map_frame:
            T = self.cam_to_map()
            if T is None:
                return              # fail safe: no pose without a valid map frame
            pos, R, frame = T[0].apply(pos) + T[1], T[0] * R, a.map_frame

        msg = PoseStamped()
        msg.header = header
        msg.header.frame_id = frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pos)
        q = R.as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y = float(q[0]), float(q[1])
        msg.pose.orientation.z, msg.pose.orientation.w = float(q[2]), float(q[3])
        self.pub.publish(msg)
        self.pub_pred.publish(Bool(data=bool(predicted)))
        if self.pub_marker is not None:
            self.publish_marker(msg, predicted)

    def publish_marker(self, pose_msg, predicted):
        """
        Box CUBE marker. Stamp 0 (latest TF) and no lifetime, so it does not
        depend on the clock of the machine running RViz; delete_marker()
        removes it when tracking is lost.
        """
        a = self.args
        m = Marker()
        m.header.frame_id = pose_msg.header.frame_id
        m.ns, m.id = "box", 0
        m.type, m.action = Marker.CUBE, Marker.ADD
        m.pose = pose_msg.pose
        m.scale.x, m.scale.y, m.scale.z = a.box_x, a.box_y, a.box_z
        m.color.r, m.color.g, m.color.b = (1.0, 0.6, 0.0) if predicted else (0.1, 0.9, 0.2)
        m.color.a = 0.5
        self.pub_marker.publish(MarkerArray(markers=[m]))
        self.marker_shown = True

    def delete_marker(self):
        if self.pub_marker is None or not self.marker_shown:
            return
        m = Marker()
        m.ns, m.id, m.action = "box", 0, Marker.DELETE
        self.pub_marker.publish(MarkerArray(markers=[m]))
        self.marker_shown = False

    def draw_status(self, img, meas, predicted, publish, t_now):
        if publish:
            pos, R = self.kf.state()
            rv = R.as_rotvec()
            cv2.drawFrameAxes(img, self.K, self.dist, rv, pos, 0.15, 4)
        if not publish:
            txt, col = "NO PUBLISH (lost)", (0, 0, 255)
        elif predicted:
            dt = (t_now - self.last_meas_t) * 1e3
            txt, col = f"PREDICT  {dt:.0f} ms since meas", (0, 200, 255)
        else:
            sd = np.linalg.norm(self.kf.pos_std()) * 1e3
            txt = (f"MEAS ids={meas['ids']} rec={meas['rec']} rms={meas['rms']:.2f}px "
                   f"depth={meas['n_depth']} sd={sd:.1f}mm")
            if self.depth_bias is not None:
                txt += f" dbias={self.depth_bias * 1e3:+.0f}mm"
            col = (0, 255, 0)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if self.learner is not None:
            cv2.rectangle(img, (0, 40), (img.shape[1], 76), (0, 0, 0), -1)
            cv2.putText(img, f"LEARN {self.learner.progress() or '(no new marker seen)'}",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Publish /box_tf (box center PoseStamped) from ArUco")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--topic", default="/box_tf")
    ap.add_argument("--cam-frame", default="top_azure_rgb_link")
    ap.add_argument("--box-frame", default="box", help="child frame of the camera->box TF")
    ap.add_argument("--no-tf", action="store_true", help="do not broadcast camera->box TF")
    ap.add_argument("--map-frame", default="base_link",
                    help="publish /box_tf and the marker in this frame (looked up from TF), "
                         "e.g. base_link or map; default: camera frame")
    ap.add_argument("--no-marker", action="store_true", help="do not publish <topic>/marker")
    ap.add_argument("--dictionary", choices=["4X4_50", "5X5_100", "6X6_250", "7X7_1000"],
                    default="6X6_250")
    # geometry [m]
    ap.add_argument("--marker-length", type=float, default=0.08)
    ap.add_argument("--box-x", type=float, default=0.310, help="front-back size")
    ap.add_argument("--box-y", type=float, default=0.410, help="left-right size")
    ap.add_argument("--box-z", type=float, default=0.280, help="height")
    ap.add_argument("--edge-x", type=float, default=0.050,
                    help="marker center distance from front/back edge (along x)")
    ap.add_argument("--edge-y", type=float, default=0.045,
                    help="marker center distance from left/right edge (along y)")
    ap.add_argument("--marker-yaw", nargs="*", default=[],
                    help="per-marker mounting yaw fix, e.g. 2:90 3:180 (deg about box +z)")
    ap.add_argument("--extra-markers",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_extra_markers.yaml"),
                    help="YAML with learned markers on other faces (loaded if it exists)")
    ap.add_argument("--learn", action="store_true",
                    help="learn unknown markers (e.g. front face) while the top face is visible")
    ap.add_argument("--learn-ids", type=int, nargs="*", default=[],
                    help="ids to learn (default: any unknown id)")
    ap.add_argument("--learn-samples", type=int, default=150,
                    help="frames per marker before it is solved and saved")
    # measurement
    ap.add_argument("--corner-refine", choices=["none", "subpix", "contour", "apriltag"],
                    default="contour",
                    help="ArUco corner refinement (contour: least jitter and most detections "
                         "in tests; apriltag is too slow for 30 fps)")
    ap.add_argument("--depth-mode", default="OFF",
                    choices=["OFF", "NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_2X2BINNED"],
                    help="fuse Azure Kinect depth (experimental: steadier when the box is still, "
                         "no gain while it moves; e.g. NFOV_UNBINNED)")
    ap.add_argument("--depth-sigma", type=float, default=0.005, help="per-sample depth noise [m]")
    ap.add_argument("--depth-grid", type=int, default=6, help="depth samples per face = grid^2")
    ap.add_argument("--depth-margin", type=float, default=0.8,
                    help="sampled part of each face (fraction of half size, avoids edges)")
    ap.add_argument("--depth-gate", type=float, default=0.03,
                    help="drop depth samples farther than this from the RGB pose [m]")
    ap.add_argument("--depth-bias-tau", type=float, default=10.0,
                    help="time constant [s] for tracking the depth-vs-RGB offset; 0 = trust "
                         "depth's absolute distance (only if a tape measure confirms it)")
    ap.add_argument("--depth-max-speed", type=float, default=0.05,
                    help="use depth only while the box moves slower than this [m/s]")
    ap.add_argument("--depth-max-rot-speed", type=float, default=0.3,
                    help="... and rotates slower than this [rad/s]")
    ap.add_argument("--depth-min-cos", type=float, default=0.3,
                    help="skip faces seen at a grazing angle (cos of view angle)")
    ap.add_argument("--pix-sigma", type=float, default=1.0,
                    help="min corner noise std [px] for the PnP covariance")
    ap.add_argument("--init-px", type=float, default=15.0,
                    help="a marker agrees with an init hypothesis below this RMS [px]")
    ap.add_argument("--huber-px", type=float, default=2.0, help="Huber threshold on corner residual [px]")
    ap.add_argument("--outlier-px", type=float, default=5.0,
                    help="drop a marker whose RMS exceeds max(this, 3x median of the others) [px]")
    ap.add_argument("--max-rms", type=float, default=4.0, help="max reprojection RMS [px]")
    ap.add_argument("--r-pos", type=float, default=0.001,
                    help="measurement std floor for position (model error) [m]")
    ap.add_argument("--r-rot", type=float, default=0.3,
                    help="measurement std floor for rotation (model error) [deg]")
    # recovery of blurred markers
    ap.add_argument("--no-recover", action="store_true",
                    help="disable matching undecoded quads to the predicted markers")
    ap.add_argument("--recover-frac", type=float, default=0.25,
                    help="max mean corner distance to prediction, as fraction of marker side")
    ap.add_argument("--recover-window", type=float, default=0.5,
                    help="recover only this long after the last decoded marker [s]")
    ap.add_argument("--recover-sigma", type=float, default=2.0,
                    help="corner noise multiplier for recovered markers")
    # Kalman / gating
    ap.add_argument("--q-pos", type=float, default=0.1,
                    help="pos accel noise [m^2/s^3]; higher follows faster but passes more depth noise")
    ap.add_argument("--q-rot", type=float, default=3.0, help="rot accel noise [rad^2/s^3]")
    ap.add_argument("--max-predict", type=float, default=0.2,
                    help="stop publishing this long after last accepted measurement [s]")
    ap.add_argument("--reinit-after", type=float, default=0.5,
                    help="reset filter if no measurement for this long [s]")
    ap.add_argument("--gate", type=float, default=30.0,
                    help="outlier gate, chi-square 6 DOF (99.9%% = 22.5)")
    ap.add_argument("--soft-gate", type=float, default=8.0,
                    help="above this chi-square distance the measurement covariance is inflated "
                         "by d2/soft_gate (robust update against single-frame spikes)")
    ap.add_argument("--max-reject", type=int, default=3,
                    help="after this many consecutive rejections, re-initialize on the measurement")
    ap.add_argument("--raw", action="store_true",
                    help="publish raw measurement when available (Kalman only fills gaps)")
    ap.add_argument("--check", action="store_true", help="log per-marker yaw vs ID4")
    add_exposure_args(ap)
    ap.add_argument("--no-display", action="store_true")
    return ap


def main():
    args = build_arg_parser().parse_args()

    rclpy.init(args=None)
    node = BoxTFPublisher(args)
    try:
        node.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
