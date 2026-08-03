#!/usr/bin/env python3
"""Estimate one rigid object's pose from multiple ArUco markers.

Object model
------------
- Cuboid size: 0.05 m x 0.05 m x 0.10 m
- Marker size: 0.04 m x 0.04 m
- One marker is attached to each of the four vertical side faces.
- Two markers are in the upper half and two are in the lower half.
- The upper/lower placement alternates around the four side faces.

Unlike averaging raw marker positions or independently estimated marker poses,
this script registers the 3-D location of every marker corner in the object
frame and solves one joint PnP problem using all visible registered corners.

IMPORTANT
---------
The default marker layout below assumes IDs 0, 1, 2, 3. Change
DEFAULT_MARKER_LAYOUT or provide --layout-yaml so that the IDs, faces,
upper/lower locations, and in-plane rotations exactly match the real object.
"""

import argparse
import time
from collections import deque
from math import sqrt
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker, MarkerArray


# =============================================================================
# Physical dimensions [m]
# =============================================================================
OBJECT_SIZE_X = 0.05
OBJECT_SIZE_Y = 0.05
OBJECT_SIZE_Z = 0.10
DEFAULT_MARKER_LENGTH = 0.04

UPPER_MARKER_Z = +OBJECT_SIZE_Z / 4.0
LOWER_MARKER_Z = -OBJECT_SIZE_Z / 4.0


# =============================================================================
# Default marker layout
# =============================================================================
# Object frame:
#   +x: right
#   +y: front -> back
#   +z: upward
#
# Face names:
#   +x/right, -x/left, +y/back, -y/front
#
# rotation_deg:
#   Positive means counter-clockwise when looking directly at the printed QR
#   from outside the object. Set this to 90, 180, or 270 if a printed QR is
#   rotated on its face.
#
# This default realizes an alternating upper/lower arrangement around the four
# side faces. Replace the IDs and face assignment with the real arrangement.
DEFAULT_MARKER_LAYOUT = {
    0: {"face": "-x", "level": "lower", "rotation_deg": 0.0},
    1: {"face": "-y", "level": "upper", "rotation_deg": 0.0},
    2: {"face": "+x", "level": "lower", "rotation_deg": 0.0},
    3: {"face": "+y", "level": "upper", "rotation_deg": 0.0},
}


# =============================================================================
# Transform and quaternion utilities
# =============================================================================
def rotation_matrix_to_quaternion(matrix):
    """Return an [x, y, z, w] quaternion for a 3x3 rotation matrix."""
    matrix = np.asarray(matrix, dtype=float)
    q = np.zeros(4, dtype=float)
    trace = np.trace(matrix)

    if trace > 0.0:
        s = 0.5 / sqrt(trace + 1.0)
        q[:] = [
            (matrix[2, 1] - matrix[1, 2]) * s,
            (matrix[0, 2] - matrix[2, 0]) * s,
            (matrix[1, 0] - matrix[0, 1]) * s,
            0.25 / s,
        ]
    else:
        index = int(np.argmax(np.diag(matrix)))

        if index == 0:
            s = 2.0 * sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            )
            q[:] = [
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[2, 1] - matrix[1, 2]) / s,
            ]
        elif index == 1:
            s = 2.0 * sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            )
            q[:] = [
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
            ]
        else:
            s = 2.0 * sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            )
            q[:] = [
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]

    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def make_transform(rotation_matrix, translation):
    """Create a homogeneous transform T_A_B from R_A_B and p_A_B."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def quaternion_average_markley(quaternions, weights=None):
    """Average [x, y, z, w] quaternions using the Markley method."""
    if not quaternions:
        raise ValueError("At least one quaternion is required")

    quaternions = [
        np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
        for quaternion in quaternions
    ]

    if weights is None:
        weights = np.ones(len(quaternions), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != len(quaternions):
        raise ValueError("weights and quaternions must have equal lengths")

    accumulator = np.zeros((4, 4), dtype=float)
    for weight, quaternion in zip(weights, quaternions):
        accumulator += float(weight) * np.outer(quaternion, quaternion)

    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, int(np.argmax(eigenvalues))]

    # Keep a stable sign for temporal filtering and ROS output.
    if result[3] < 0.0:
        result = -result

    return result / np.linalg.norm(result)


# =============================================================================
# Layout construction
# =============================================================================
def normalize_face_name(face):
    aliases = {
        "+x": "+x",
        "x+": "+x",
        "right": "+x",
        "-x": "-x",
        "x-": "-x",
        "left": "-x",
        "+y": "+y",
        "y+": "+y",
        "back": "+y",
        "rear": "+y",
        "-y": "-y",
        "y-": "-y",
        "front": "-y",
    }

    normalized = str(face).strip().lower()
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported face '{face}'. Use +x, -x, +y, -y, "
            "right, left, back, or front."
        )
    return aliases[normalized]


def level_to_z(level):
    normalized = str(level).strip().lower()
    if normalized in {"upper", "top", "up"}:
        return UPPER_MARKER_Z
    if normalized in {"lower", "bottom", "down"}:
        return LOWER_MARKER_Z
    raise ValueError(f"Unsupported marker level '{level}'. Use upper or lower.")


def face_geometry(face, marker_z):
    """Return center, marker-right, marker-up, and outward normal in object frame."""
    half_x = OBJECT_SIZE_X / 2.0
    half_y = OBJECT_SIZE_Y / 2.0
    face = normalize_face_name(face)

    if face == "+x":
        center = np.array([half_x, 0.0, marker_z], dtype=float)
        right = np.array([0.0, 1.0, 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
    elif face == "-x":
        center = np.array([-half_x, 0.0, marker_z], dtype=float)
        right = np.array([0.0, -1.0, 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    elif face == "+y":
        center = np.array([0.0, half_y, marker_z], dtype=float)
        right = np.array([-1.0, 0.0, 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.array([0.0, 1.0, 0.0], dtype=float)
    else:  # -y
        center = np.array([0.0, -half_y, marker_z], dtype=float)
        right = np.array([1.0, 0.0, 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.array([0.0, -1.0, 0.0], dtype=float)

    return center, right, up, normal


def build_marker_model(marker_id, config, marker_length):
    """Build one marker's exact 3-D geometry in the object frame."""
    face = normalize_face_name(config["face"])

    if "z" in config:
        marker_z = float(config["z"])
    elif "z_center" in config:
        marker_z = float(config["z_center"])
    else:
        marker_z = level_to_z(config.get("level", "upper"))

    rotation_deg = float(config.get("rotation_deg", 0.0))
    center, right, up, normal = face_geometry(face, marker_z)

    # Rotate the printed marker within the side face. Positive angle is CCW
    # when looking at the QR from outside the object.
    in_plane_rotation = Rotation.from_rotvec(
        np.deg2rad(rotation_deg) * normal
    ).as_matrix()
    right = in_plane_rotation @ right
    up = in_plane_rotation @ up

    # Numerical safety and consistency.
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    normal = np.cross(right, up)
    normal /= np.linalg.norm(normal)

    marker_half = marker_length / 2.0

    # Must match cv2.aruco corner ordering:
    # top-left, top-right, bottom-right, bottom-left.
    corners_object = np.asarray(
        [
            center - marker_half * right + marker_half * up,
            center + marker_half * right + marker_half * up,
            center + marker_half * right - marker_half * up,
            center - marker_half * right - marker_half * up,
        ],
        dtype=np.float32,
    )

    # Columns are marker-frame axes represented in the object frame.
    rotation_object_marker = np.column_stack((right, up, normal))
    object_to_marker = make_transform(rotation_object_marker, center)
    marker_to_object = np.linalg.inv(object_to_marker)

    return {
        "id": int(marker_id),
        "face": face,
        "z": marker_z,
        "rotation_deg": rotation_deg,
        "center_object": center,
        "corners_object": corners_object,
        "object_to_marker": object_to_marker,
        "marker_to_object": marker_to_object,
    }


def load_marker_layout(path, marker_length):
    """Load marker layout from YAML or use DEFAULT_MARKER_LAYOUT."""
    if path:
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        if not isinstance(data, dict):
            raise ValueError("Marker layout YAML must contain a mapping")

        layout_data = data.get("markers", data)
    else:
        layout_data = DEFAULT_MARKER_LAYOUT

    if not isinstance(layout_data, dict) or not layout_data:
        raise ValueError("Marker layout must contain at least one marker")

    layout = {}
    for marker_id_raw, config in layout_data.items():
        marker_id = int(marker_id_raw)
        if not isinstance(config, dict):
            raise ValueError(f"Marker ID {marker_id} config must be a mapping")
        if "face" not in config:
            raise ValueError(f"Marker ID {marker_id} is missing 'face'")

        layout[marker_id] = build_marker_model(
            marker_id,
            config,
            marker_length,
        )

    return layout


# =============================================================================
# Calibration loading
# =============================================================================
def load_base_to_camera(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    translation = np.asarray(data["translation"], dtype=float)

    if "quaternion" in data:
        quaternion = np.asarray(data["quaternion"], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
    elif "rotation_matrix" in data:
        quaternion = rotation_matrix_to_quaternion(
            np.asarray(data["rotation_matrix"], dtype=float)
        )
    else:
        raise KeyError(
            "Calibration YAML needs either 'quaternion' or 'rotation_matrix'"
        )

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = translation
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return transform, quaternion


def load_intrinsics(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    return (
        np.asarray(data["camera_matrix"], dtype=np.float32),
        np.asarray(data["dist_coeff"], dtype=np.float32),
    )


# =============================================================================
# Temporal filtering
# =============================================================================
class PoseFilter:
    def __init__(self, size):
        self.samples = deque(maxlen=max(1, int(size)))

    def clear(self):
        self.samples.clear()

    def update(self, position, quaternion):
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
        quaternion /= np.linalg.norm(quaternion)

        if self.samples and np.dot(quaternion, self.samples[-1][1]) < 0.0:
            quaternion = -quaternion

        self.samples.append((position.copy(), quaternion.copy()))

        position_out = np.mean(
            [sample[0] for sample in self.samples],
            axis=0,
        )
        quaternion_out = quaternion_average_markley(
            [sample[1] for sample in self.samples]
        )

        if np.dot(quaternion_out, quaternion) < 0.0:
            quaternion_out = -quaternion_out

        return position_out, quaternion_out


# =============================================================================
# ROS 2 node
# =============================================================================
class JointPnPObjectPublisher(Node):
    def __init__(self, args):
        super().__init__("joint_pnp_object_pose_publisher")
        self.args = args

        self.base_to_camera, camera_quaternion = load_base_to_camera(args.yaml)
        self.marker_layout = load_marker_layout(
            args.layout_yaml,
            args.marker_length,
        )

        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            args.pose_topic,
            10,
        )

        # Transient-local keeps the latest MarkerArray for RViz instances that
        # start after this node. Reliable transport is useful across two PCs.
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            args.marker_topic,
            marker_qos,
        )

        self.publish_static_camera_tf(camera_quaternion)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            args.width,
            args.height,
            rs.format.bgr8,
            args.fps,
        )
        self.pipeline.start(config)

        if args.intrinsics:
            self.camera_matrix, self.dist_coeffs = load_intrinsics(
                args.intrinsics
            )
        else:
            self.camera_matrix = None
            self.dist_coeffs = None

        dictionary_id = getattr(cv2.aruco, "DICT_" + args.dictionary)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

        self.detector_parameters = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create()
        )

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.dictionary,
                self.detector_parameters,
            )
        else:
            self.aruco_detector = None

        self.marker_local_corners = self.make_marker_local_corners(
            args.marker_length
        )
        self.pose_filter = PoseFilter(args.filter_N)
        self.last_valid_detection_time = None
        self.previous_camera_to_object = None

        self.display = not args.no_display
        if self.display:
            cv2.namedWindow("Joint object pose", cv2.WINDOW_NORMAL)

        layout_description = ", ".join(
            f"ID {marker_id}: {model['face']} z={model['z']:+.3f} m "
            f"rot={model['rotation_deg']:.1f} deg"
            for marker_id, model in sorted(self.marker_layout.items())
        )
        self.get_logger().info(f"Registered marker layout: {layout_description}")
        self.get_logger().info(
            f"Publishing {args.base_frame} -> {args.object_frame}, "
            f"PoseStamped on {args.pose_topic}, MarkerArray on {args.marker_topic}"
        )

    @staticmethod
    def make_marker_local_corners(marker_length):
        marker_half = marker_length / 2.0
        return np.asarray(
            [
                [-marker_half, marker_half, 0.0],
                [marker_half, marker_half, 0.0],
                [marker_half, -marker_half, 0.0],
                [-marker_half, -marker_half, 0.0],
            ],
            dtype=np.float32,
        )

    def publish_static_camera_tf(self, quaternion):
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.args.base_frame
        message.child_frame_id = self.args.cam_frame

        position = self.base_to_camera[:3, 3]
        message.transform.translation.x = float(position[0])
        message.transform.translation.y = float(position[1])
        message.transform.translation.z = float(position[2])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])

        self.static_broadcaster.sendTransform(message)

    def get_intrinsics_from_camera(self):
        profile = self.pipeline.get_active_profile()
        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self.camera_matrix = np.asarray(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.asarray(
            intrinsics.coeffs[:5],
            dtype=np.float32,
        )

        self.get_logger().info("Using color intrinsics reported by RealSense")

    def detect_registered_markers(self, image):
        """Detect markers and retain only IDs registered in marker_layout."""
        if self.camera_matrix is None:
            self.get_intrinsics_from_camera()

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(image)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                image,
                self.dictionary,
                parameters=self.detector_parameters,
            )

        if ids is None or len(corners) == 0:
            return []

        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        detections = []

        for marker_corners, marker_id_array in zip(corners, ids):
            marker_id = int(marker_id_array[0])
            image_points = np.asarray(
                marker_corners,
                dtype=np.float32,
            ).reshape(4, 2)
            center = np.mean(image_points, axis=0).astype(int)

            if marker_id not in self.marker_layout:
                cv2.putText(
                    image,
                    f"ID {marker_id}: ignored",
                    (int(center[0]), int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
                continue

            detections.append(
                {
                    "id": marker_id,
                    "image_points": image_points,
                }
            )
            cv2.putText(
                image,
                f"ID {marker_id}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        return detections

    def estimate_single_marker_object_pose(self, detection):
        """Estimate T_camera_object from one registered marker."""
        marker_id = detection["id"]
        image_points = detection["image_points"]

        pnp_flag = (
            cv2.SOLVEPNP_IPPE_SQUARE
            if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
            else cv2.SOLVEPNP_ITERATIVE
        )

        success, rvec, tvec = cv2.solvePnP(
            self.marker_local_corners,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=pnp_flag,
        )

        if not success and pnp_flag != cv2.SOLVEPNP_ITERATIVE:
            success, rvec, tvec = cv2.solvePnP(
                self.marker_local_corners,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            return None

        camera_to_marker = make_transform(
            cv2.Rodrigues(rvec)[0],
            tvec.reshape(3),
        )
        camera_to_object = (
            camera_to_marker
            @ self.marker_layout[marker_id]["marker_to_object"]
        )

        return camera_to_object

    def collect_joint_correspondences(self, detections):
        object_points = []
        image_points = []

        for detection in detections:
            marker_id = detection["id"]
            object_points.append(
                self.marker_layout[marker_id]["corners_object"]
            )
            image_points.append(detection["image_points"])

        return (
            np.concatenate(object_points, axis=0).astype(np.float32),
            np.concatenate(image_points, axis=0).astype(np.float32),
        )

    def solve_joint_pnp(self, detections):
        """Solve one camera-to-object pose using all visible marker corners."""
        object_points, image_points = self.collect_joint_correspondences(
            detections
        )

        ransac_flag = (
            cv2.SOLVEPNP_EPNP
            if hasattr(cv2, "SOLVEPNP_EPNP")
            else cv2.SOLVEPNP_ITERATIVE
        )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            iterationsCount=self.args.ransac_iterations,
            reprojectionError=self.args.ransac_reprojection_error,
            confidence=self.args.ransac_confidence,
            flags=ransac_flag,
        )

        if not success or inliers is None or len(inliers) < 4:
            # A non-RANSAC fallback is useful when only a small number of
            # perfectly valid points is available.
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            inliers = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)

        if not success:
            return None

        inlier_indices = inliers.reshape(-1)
        if (
            hasattr(cv2, "solvePnPRefineLM")
            and len(inlier_indices) >= 4
        ):
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=object_points[inlier_indices],
                imagePoints=image_points[inlier_indices],
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                rvec=rvec,
                tvec=tvec,
            )

        camera_to_object = make_transform(
            cv2.Rodrigues(rvec)[0],
            tvec.reshape(3),
        )

        # Reject mirrored/behind-camera solutions.
        points_camera = (
            camera_to_object[:3, :3] @ object_points.T
            + camera_to_object[:3, 3:4]
        ).T
        if np.any(points_camera[:, 2] <= 0.0):
            return None

        return camera_to_object

    def marker_reprojection_errors(self, camera_to_object, detections):
        rotation_vector, _ = cv2.Rodrigues(camera_to_object[:3, :3])
        translation_vector = camera_to_object[:3, 3].reshape(3, 1)
        errors = {}

        for detection in detections:
            marker_id = detection["id"]
            projected_points, _ = cv2.projectPoints(
                self.marker_layout[marker_id]["corners_object"],
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                self.dist_coeffs,
            )
            projected_points = projected_points.reshape(4, 2)
            corner_errors = np.linalg.norm(
                projected_points - detection["image_points"],
                axis=1,
            )
            errors[marker_id] = float(np.mean(corner_errors))

        return errors

    def estimate_fallback_pose(self, detections):
        """Fuse per-marker object poses only when joint PnP is unavailable."""
        candidates = []
        weights = []

        for detection in detections:
            candidate = self.estimate_single_marker_object_pose(detection)
            if candidate is None:
                continue

            errors = self.marker_reprojection_errors(candidate, [detection])
            error = errors[detection["id"]]
            weight = 1.0 / max(error, 0.25) ** 2
            candidates.append(candidate)
            weights.append(weight)

        if not candidates:
            return None

        weights = np.asarray(weights, dtype=float)
        weights /= np.sum(weights)

        position = np.sum(
            [weight * candidate[:3, 3] for weight, candidate in zip(weights, candidates)],
            axis=0,
        )
        quaternion = quaternion_average_markley(
            [Rotation.from_matrix(candidate[:3, :3]).as_quat() for candidate in candidates],
            weights,
        )

        return make_transform(
            Rotation.from_quat(quaternion).as_matrix(),
            position,
        )

    def estimate_object_pose(self, detections):
        """Estimate pose, reject bad markers, and return diagnostics."""
        if not detections:
            return None

        if len(detections) == 1:
            camera_to_object = self.estimate_single_marker_object_pose(
                detections[0]
            )
        else:
            camera_to_object = self.solve_joint_pnp(detections)

        if camera_to_object is None:
            camera_to_object = self.estimate_fallback_pose(detections)

        if camera_to_object is None:
            return None

        errors = self.marker_reprojection_errors(
            camera_to_object,
            detections,
        )

        good_detections = [
            detection
            for detection in detections
            if errors[detection["id"]]
            <= self.args.max_marker_reprojection_error
        ]

        # Re-solve once after removing a whole marker whose four corners do not
        # agree with the rigid object model.
        if 0 < len(good_detections) < len(detections):
            rejected_ids = sorted(
                set(detection["id"] for detection in detections)
                - set(detection["id"] for detection in good_detections)
            )
            self.get_logger().warning(
                f"Rejecting marker IDs with large reprojection error: {rejected_ids}"
            )

            if len(good_detections) == 1:
                refined_pose = self.estimate_single_marker_object_pose(
                    good_detections[0]
                )
            else:
                refined_pose = self.solve_joint_pnp(good_detections)

            if refined_pose is not None:
                camera_to_object = refined_pose
                detections = good_detections
                errors = self.marker_reprojection_errors(
                    camera_to_object,
                    detections,
                )

        used_ids = [detection["id"] for detection in detections]
        rms_error = float(
            np.sqrt(np.mean(np.square(list(errors.values()))))
        )

        if rms_error > self.args.max_pose_reprojection_error:
            self.get_logger().warning(
                f"Rejecting object pose: RMS reprojection error "
                f"{rms_error:.2f} px exceeds "
                f"{self.args.max_pose_reprojection_error:.2f} px"
            )
            return None

        return camera_to_object, used_ids, rms_error, errors

    @staticmethod
    def fill_pose(pose, position, quaternion):
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])

    def publish_result(self, position, quaternion):
        stamp = self.get_clock().now().to_msg()

        # Dynamic TF: base -> object center.
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.args.base_frame
        transform.child_frame_id = self.args.object_frame
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(transform)

        # Object-center PoseStamped.
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.args.base_frame
        self.fill_pose(pose.pose, position, quaternion)
        self.pose_publisher.publish(pose)

        # RViz MarkerArray. A zero stamp asks RViz to use the latest available
        # transform and avoids cross-PC clock skew when Fixed Frame != base.
        marker = Marker()
        marker.header.frame_id = self.args.base_frame
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        marker.ns = "detected_object"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        self.fill_pose(marker.pose, position, quaternion)
        marker.scale.x = OBJECT_SIZE_X
        marker.scale.y = OBJECT_SIZE_Y
        marker.scale.z = OBJECT_SIZE_Z
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.color.a = 0.8
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def draw_object_axes(self, image, camera_to_object):
        rvec, _ = cv2.Rodrigues(camera_to_object[:3, :3])
        tvec = camera_to_object[:3, 3].reshape(3, 1)
        cv2.drawFrameAxes(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            0.05,
        )

    def run(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.001)

                frame = self.pipeline.wait_for_frames().get_color_frame()
                if not frame:
                    continue

                image = np.asanyarray(frame.get_data()).copy()
                detections = self.detect_registered_markers(image)
                result = self.estimate_object_pose(detections)

                if result is not None:
                    camera_to_object, used_ids, rms_error, per_marker_errors = result
                    self.previous_camera_to_object = camera_to_object.copy()
                    self.last_valid_detection_time = time.monotonic()

                    self.draw_object_axes(image, camera_to_object)

                    base_to_object = self.base_to_camera @ camera_to_object
                    position = base_to_object[:3, 3]
                    quaternion = Rotation.from_matrix(
                        base_to_object[:3, :3]
                    ).as_quat()
                    position, quaternion = self.pose_filter.update(
                        position,
                        quaternion,
                    )
                    self.publish_result(position, quaternion)

                    error_text = ", ".join(
                        f"{marker_id}:{per_marker_errors[marker_id]:.1f}px"
                        for marker_id in used_ids
                    )
                    cv2.putText(
                        image,
                        f"IDs {used_ids} | RMS {rms_error:.2f}px",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        error_text,
                        (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                else:
                    if (
                        self.last_valid_detection_time is not None
                        and time.monotonic() - self.last_valid_detection_time
                        > self.args.filter_reset_timeout
                    ):
                        self.pose_filter.clear()
                        self.previous_camera_to_object = None
                        self.last_valid_detection_time = None

                if self.display:
                    cv2.putText(
                        image,
                        "q: quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Joint object pose", image)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


# =============================================================================
# Command-line arguments
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a cuboid pose from registered ArUco corners using joint PnP "
            "and publish TF, PoseStamped, and MarkerArray"
        )
    )

    parser.add_argument("--yaml", default="TF_base2cam.yaml")
    parser.add_argument("--intrinsics", default="")
    parser.add_argument(
        "--layout-yaml",
        default="",
        help=(
            "Optional marker-layout YAML. Without it, DEFAULT_MARKER_LAYOUT "
            "inside this script is used."
        ),
    )

    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--cam-frame", default="rs_camera")
    parser.add_argument("--object-frame", default="object_frame")
    parser.add_argument("--pose-topic", default="/object_pose")
    parser.add_argument("--marker-topic", default="/qr_marker_array")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--marker-length",
        type=float,
        default=DEFAULT_MARKER_LENGTH,
    )
    parser.add_argument(
        "--dictionary",
        default="6X6_250",
        choices=["4X4_50", "5X5_100", "6X6_250", "7X7_1000"],
    )

    parser.add_argument("--filter-N", type=int, default=10)
    parser.add_argument(
        "--filter-reset-timeout",
        type=float,
        default=0.5,
        help="Clear the temporal filter after this many seconds without a valid pose",
    )

    parser.add_argument("--ransac-iterations", type=int, default=200)
    parser.add_argument(
        "--ransac-reprojection-error",
        type=float,
        default=3.0,
        help="Point-level RANSAC reprojection threshold [pixel]",
    )
    parser.add_argument("--ransac-confidence", type=float, default=0.99)
    parser.add_argument(
        "--max-marker-reprojection-error",
        type=float,
        default=4.0,
        help="Reject an entire marker above this mean corner error [pixel]",
    )
    parser.add_argument(
        "--max-pose-reprojection-error",
        type=float,
        default=5.0,
        help="Reject the final pose above this RMS marker error [pixel]",
    )

    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init(args=None)
    node = JointPnPObjectPublisher(args)

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
