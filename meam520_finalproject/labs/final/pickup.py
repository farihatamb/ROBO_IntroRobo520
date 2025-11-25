import numpy as np
from math import pi
from lib.calculateFK import FK
from lib.calculateIK6 import IK
from core.interfaces import ArmController


def pick_up_block(H_wb, arm: ArmController, fk):
    # this should move to q1 which is a z offset above the pick_up_block
    # then move down to the appropriate height to pick up the block (q2)
    # then close gripper
    # then lift arm (q3)
    #this offset is to grasp the top quarter of the block
    H_wb[2, 3]=H_wb[2, 3]-0.001
    #Homography from world to block frame with z offset
    H_wboffset=H_wb.copy()
    H_wboffset[2, 3]=H_wboffset[2, 3]+.10
    print("----")
    print(H_wb)
    print(H_wboffset)
    #homography from world to end effector
    [_,H_we] = fk.forward(arm.get_positions())
    #find H from end effector to some height above block
    rot=H_wboffset[0:3, 0:3]@np.array([[1, 0, 0], [0, np.cos(pi), -np.sin(pi)], [0, np.sin(pi), np.cos(pi)]])
    trans=H_wboffset[:3, 3]
    #shift translation up in the z axis
    target1 = {'R':rot,'t': trans}
    ik = IK()
    q1 = ik.panda_ik(target1)
    #this translation should be suitible for grasping
    rot=H_wb[0:3, 0:3]@np.array([[1, 0, 0], [0, np.cos(pi), -np.sin(pi)], [0, np.sin(pi), np.cos(pi)]])
    trans=H_wb[:3, 3]
    target2 = {'R':rot,'t': trans}
    q2 = ik.panda_ik(target2)
    #open gripper
    #we should probably move to H with a shift in the z axis
    arm.safe_move_to_position(q1[0])
    arm.exec_gripper_cmd(3, 10)
    # then move arm downwards
    arm.safe_move_to_position(q2[0])
    arm.exec_gripper_cmd(0.045, 25)
    #close gripper
    #then back up
    arm.safe_move_to_position(q1[0])
