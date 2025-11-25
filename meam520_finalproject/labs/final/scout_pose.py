"""Return Scout Poses"""
import numpy as np
from numpy import pi
from core.interfaces import ArmController

def get_static_scout_pose(team):
    mult = 1 if team=='blue' else -1
    scout_pose = np.array(
        [
            mult*pi/10,
            0.0,
            0.0,
            -pi/2,
            0.0,
            pi/2,
            pi/4 + mult*pi/10
        ]
    )
    return scout_pose

def get_dynamic_scout_pose(team):
    multiplier = -1 if team=='blue' else 1
    flex_amount = pi/6
    scout_pose = np.array(
        [
            multiplier * pi/2,
            flex_amount*(2/3),
            0.0,
            (-pi/2)+(flex_amount*4/3),
            0.0,
            pi/2-(flex_amount/6),
            pi/4
        ]
    )
    return scout_pose

def move_to_dynamic_scout_pose(arm:ArmController, team):
    """
    Params:
        arm: ArmController
    """
    arm.safe_move_to_position(get_dynamic_scout_pose(team))

def move_to_static_scout_pose(arm:ArmController, team):
    """
    Params:
        arm: ArmController
    """
    arm.safe_move_to_position(get_static_scout_pose(team))