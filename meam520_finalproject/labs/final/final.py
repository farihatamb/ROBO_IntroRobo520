import sys
import time
import numpy as np
from copy import deepcopy
from math import pi
from lib.calculateFK import FK
from lib.calculateIK6 import IK

import rospy
import tf
tf_broad  = tf.TransformBroadcaster()
# Broadcasts a T0e as the transform from given frame to world frame
def show_pose(T0e,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(T0e),
        tf.transformations.quaternion_from_matrix(T0e),
        rospy.Time.now(),
        frame,
        "world"
    )

def show_target(target):
    T0_target = np.vstack((np.hstack((target['R'], np.array([target['t']]).T)), np.array([[0, 0, 0, 1]])))
    show_pose(T0_target,"target")

def show_frame(frame, name):
    show_pose(frame, name)

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

from scout_pose import move_to_static_scout_pose, move_to_dynamic_scout_pose
from pickup import pick_up_block
from stack import stack_block
from block_positions import get_block_pose, get_best_pose


# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

fk = FK()
ik = IK()

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE
    """
    static_scout_pose = np.array([0.0, 0.0, 0.0, -pi/2, 0.0, pi/2, pi/4])
    [_, T_fk] = fk.forward(static_scout_pose)
    #move_to_dynamic_scout_pose(arm, team)
    trans = np.array([-0.01, -0.71, 0.35])
    from scipy.spatial.transform import Rotation

    rot = np.array([
        [ 1.23687194e-16, -1.00000000e+00, -8.65555385e-17],
        [-9.65925826e-01, -7.74688301e-17, -2.58819045e-01],
        [ 2.58819045e-01,  1.10366598e-16, -9.65925826e-01]])
    
    correction = Rotation.from_euler('ZYZ', [45, -45, 0], degrees=True)
    

    q = ik.panda_ik(({
                'R': rot.as_matrix(),
                't': trans}))
    arm.safe_move_to_position(q[0])
    [_, T_fk] = fk.forward(q[0])
    print(T_fk)

    exit()
    """
    move_to_static_scout_pose(arm, team)
    
    # Detect some blocks...

    ##STATIC BLOCKS ###
    a = [] #list of target matrix containing R and T for each block

    block_poses = detector.get_detections()
    block_transformation = list()
    for (name, pose) in block_poses:
        world_pose, pose_candidates = get_block_pose(pose, arm.get_positions(), detector, fk)
        world_pose = get_best_pose(pose_candidates, ik)
        show_frame(world_pose, name)
        print(name, '\n', pose, '\n', world_pose)
        block_transformation.append(world_pose)
    
    count=0
    #adjust buffer experimentally
    platform_floor = 0.2
    #placeholder values need adjusting
    x = 0.6
    y = 0.24
    y *= -1 if team == 'blue' else 1
    buffer = 0.025
    for staticblock in block_transformation:
        try:
            z= platform_floor+(count*.05) + buffer
            rot=np.array([[1, 0, 0], [0, np.cos(pi), -np.sin(pi)], [0, np.sin(pi), np.cos(pi)]])#@np.array([[np.cos(-pi/2), -np.sin(-pi/2), 0], [np.sin(-pi/2), np.cos(-pi/2), 0], [0, 0, 1]])
            trans=np.array([[x],[y],[z+.05]])
            #shift translation up in the z axis
            target1 = {'R':rot,'t': trans}
            show_frame(np.vstack([np.hstack([rot, trans]), np.array([0, 0, 0, 1])]), "Target")
            pick_up_block(staticblock, arm, fk)
            stack_block(x,y,z,arm,fk)
            count=count+1
        except IndexError:
            print("Block pickup failed")

    move_to_static_scout_pose(arm, team)

    block_poses = detector.get_detections()
    block_transformation = list()
    for (name, pose) in block_poses:
        world_pose, pose_candidates = get_block_pose(pose, arm.get_positions(), detector, fk)
        world_pose = get_best_pose(pose_candidates, ik)
        show_frame(world_pose, name)
        print(name, '\n', pose, '\n', world_pose)
        block_transformation.append(world_pose)

    for staticblock in block_transformation:
        try:
            z= platform_floor+(count*.05) + buffer
            rot=np.array([[1, 0, 0], [0, np.cos(pi), -np.sin(pi)], [0, np.sin(pi), np.cos(pi)]])#@np.array([[np.cos(-pi/2), -np.sin(-pi/2), 0], [np.sin(-pi/2), np.cos(-pi/2), 0], [0, 0, 1]])
            trans=np.array([[x],[y],[z+.05]])
            #shift translation up in the z axis
            target1 = {'R':rot,'t': trans}
            show_frame(np.vstack([np.hstack([rot, trans]), np.array([0, 0, 0, 1])]), "Target")
            pick_up_block(staticblock, arm, fk)
            stack_block(x,y,z,arm,fk)
            count=count+1
        except IndexError:
            print("Block Pickup failed")


    #last_cube_pose = world_pose
    #last_cube_pose[3,2] += 0.1
    #target_pose = {'R': last_cube_pose[0:3, 0:3], 't': last_cube_pose[0:3, 3]}
    #target_q = ik.panda_ik(target_pose)
    #print(target_q[0])
    #arm.safe_move_to_position(target_q[0])

    
    move_to_dynamic_scout_pose(arm, team)

    #for (name, pose) in detector.get_detections():
    #	H_0b_i = T_fk @ pose
    #    target = {'R': H_0b_i[0:3, 0:3], 't': H_0b_i[0:3, 3]}
    #    a.append(target)

    #for i in a:
    #	pick_up_block(a[0])
    # 	drop_block(i)

    #move_to_dynamic_scout_pose(arm, team)


    # Move around...

    # END STUDENT CODE
