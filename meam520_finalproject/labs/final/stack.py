import numpy as np
from math import pi
from lib.calculateFK import FK
from lib.calculateIK6 import IK
from core.interfaces import ArmController


def stack_block(x, y, z, arm: ArmController, fk):
    # xy is the location of the stack in the robot coord. frame.
    # z is how high we're gna stack
    [_,H_we] = fk.forward(arm.get_positions())
    #find H from end effector to some height above block
    #rotation from base to stack should make z point down
    rot=np.array([[1, 0, 0], [0, np.cos(pi), -np.sin(pi)], [0, np.sin(pi), np.cos(pi)]])@np.array([[np.cos(-pi/2), -np.sin(-pi/2), 0], [np.sin(-pi/2), np.cos(-pi/2), 0], [0, 0, 1]])
    trans=np.array([[x],[y],[z+.05]])
    #shift translation up in the z axis
    target1 = {'R':rot,'t': trans}
    ik = IK()
    q1 = ik.panda_ik(target1)
    if q1.shape[0] == 0:
        # No Solution
        rot = rot@np.array([[np.cos(pi), -np.sin(pi), 0], [np.sin(pi), np.cos(pi), 0], [0, 0, 1]])
        q1 = ik.panda_ik({'R':rot,'t': trans})
    #this translation should be suitible for grasping
    trans=np.array([[x],[y],[z]])
    target2 = {'R':rot,'t': trans}
    q2 = ik.panda_ik(target2)
    arm.safe_move_to_position(q1[0])
    # then move arm downwards
    arm.safe_move_to_position(q2[0])
    arm.open_gripper()
    #then back up
    arm.safe_move_to_position(q1[0])
