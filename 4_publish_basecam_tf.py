"""
Publish a static TF from <parent_frame> to <child_frame> using
the transform stored in yaml.
"""
import argparse
import yaml
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from math import atan2, asin, sqrt

def rotmat_to_quat(R):
    """3x3 rotation matrix → quaternion [x,y,z,w]."""
    q = [0,0,0,0]
    t = np.trace(R)
    if t > 0:
        s = 0.5 / sqrt(t + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2,1] - R[1,2]) * s
        q[1] = (R[0,2] - R[2,0]) * s
        q[2] = (R[1,0] - R[0,1]) * s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = 2.0 * sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            q[3] = (R[2,1] - R[1,2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = 2.0 * sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            q[3] = (R[0,2] - R[2,0]) / s
            q[0] = (R[0,1] + R[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            q[3] = (R[1,0] - R[0,1]) / s
            q[0] = (R[0,2] + R[2,0]) / s
            q[1] = (R[1,2] + R[2,1]) / s
            q[2] = 0.25 * s
    return q

def load_yaml_transform(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Prefer quaternion if present, else build it from rotation_matrix
    if 'quaternion' in data:
        quat = data['quaternion']
    elif 'rotation_matrix' in data:
        R = np.array(data['rotation_matrix'])
        quat = rotmat_to_quat(R)
    else:
        raise KeyError("YAML must contain 'quaternion' or 'rotation_matrix'")

    trans = data['translation']
    return trans, quat  # both are Python lists of length 3 / 4

def main():
    ap = argparse.ArgumentParser(description="Publish baese cam TF from YAML")
    ap.add_argument('--yaml',         default='TF_base2cam.yaml')
    ap.add_argument('--parent-frame', default='panda_link0')
    ap.add_argument('--child-frame',  default='rs_camera')
    args = ap.parse_args()

    trans, quat = load_yaml_transform(args.yaml)
    trans = [float(v) for v in trans]
    quat = [float(v) for v in quat]

    rclpy.init(args=None)
    node = Node('basecam_tf_broadcaster')
    br = tf2_ros.StaticTransformBroadcaster(node)
    tfm = TransformStamped()
    tfm.header.stamp = node.get_clock().now().to_msg()
    tfm.header.frame_id = args.parent_frame
    tfm.child_frame_id  = args.child_frame
    tfm.transform.translation.x = trans[0]
    tfm.transform.translation.y = trans[1]
    tfm.transform.translation.z = trans[2]
    tfm.transform.rotation.x = quat[0]
    tfm.transform.rotation.y = quat[1]
    tfm.transform.rotation.z = quat[2]
    tfm.transform.rotation.w = quat[3]
    br.sendTransform(tfm)

    node.get_logger().info(f"Static TF {args.parent_frame} -> {args.child_frame} published.")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
