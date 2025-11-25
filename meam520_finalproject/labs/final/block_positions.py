"""Get the positions of the blocks"""
import numpy as np
from math import pi
from core.interfaces import ObjectDetector
from lib.calculateFK import FK
from scipy.spatial.transform import Rotation
from lib.calculateFK import Axis, Transform, get_htransform_mat

def get_block_pose(pose, current_q, obj_detector:ObjectDetector, fk: FK):
    # Pose is in Camera Frame
    H_from_camera_to_ee = obj_detector.get_H_ee_camera()
    pose = H_from_camera_to_ee @ pose # Pose is now in EE frame
    # Get transformation from Robot base to EE
    _, H_to_ee = fk.forward(current_q)
    # Get tf mat from EE to Robot base
    #H_to_base = np.linalg.inv(H_to_ee)
    pose = H_to_ee @ pose
    # We now have the block pose in the world frame
    # The position is fine, we need to modify so the euler angle between the 
    # world Z and the block z is minimized.
    # The positive z can be going through any of the 6 faces of the cube, 
    # we want it going through the top
    newX = pose[:3, 0]
    newY = pose[:3, 1]
    newZ = pose[:3, 2]
    xdot = newX.dot(np.array([0,0,1]))
    ydot = newY.dot(np.array([0,0,1]))
    zdot = newZ.dot(np.array([0,0,1]))
    maxdot = np.max(np.abs([xdot, ydot, zdot]))
    if maxdot == zdot:
        # The current z is correct
        R = pose[:3,:3]
    elif maxdot == ydot:
        # Y is the correct Z
        R = pose[:3,:3] @ get_htransform_mat(Transform.ROT, Axis.X, -pi/2)[:3,:3]
    elif maxdot == xdot:
        # X is the correct Z
        R = pose[:3,:3] @ get_htransform_mat(Transform.ROT, Axis.Y, pi/2)[:3,:3]
    elif maxdot == -zdot:
        # -z is correct Z
        R = pose[:3,:3] @ get_htransform_mat(Transform.ROT, Axis.Y, -pi)[:3,:3]
    elif maxdot == -ydot:
        # -Y is the correct Z
        R = pose[:3,:3] @ get_htransform_mat(Transform.ROT, Axis.X, pi/2)[:3,:3]
    elif maxdot == -xdot:
        # -X is the correct Z
        R = pose[:3,:3] @ get_htransform_mat(Transform.ROT, Axis.Y, -pi/2)[:3,:3]
    else:
        ValueError("Something is wrong")

    R = Rotation.from_matrix(R)

    euler = R.as_euler('ZYX')
    correction = Rotation.from_euler('XYZ', np.array([-euler[2], -euler[1], 0.0]))
    R = R*correction
    # We actually only care about z, maybe
    rot_component = R.as_matrix()
    pose[:3, :3] = rot_component
    #print("Z_rot: ", correction)
    pose_candidates = [pose.copy() for _ in range(4)]
    for i in range(4):
        pose_candidates[i][:3,:3] = pose_candidates[i][:3,:3] @ get_htransform_mat(Transform.ROT, Axis.Z, i*pi/2)[:3,:3]

    return pose, pose_candidates

def get_best_pose(pose_candidates, ik):
    ik_sols = [np.inf for _ in range(4)]
    for idx, pose_candidate in enumerate(pose_candidates):
        try:
            q_sol = ik.panda_ik({
                'R': pose_candidate[0:3, 0:3]@np.array([[1, 0, 0], [0, np.cos(pi), -np.sin(pi)], [0, np.sin(pi), np.cos(pi)]]),
                't': pose_candidate[:3, 3]})
            q_sel = q_sol[0]
            ik_sols[idx] = np.linalg.norm(q_sel)
        except:
            pass  # No solution
    return pose_candidates[np.argmin(ik_sols)]