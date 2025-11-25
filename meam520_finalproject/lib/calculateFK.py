import numpy as np
from numpy import cos, sin
from math import pi
from enum import Enum

class Axis(Enum):
    X=0
    Y=1
    Z=2

class Transform(Enum):
    TRANSLATE=0
    ROT=1

# Shortcuts
X = Axis.X
Y = Axis.Y
Z = Axis.Z
TRANSLATE = Transform.TRANSLATE
ROT = Transform.ROT


def matrix_from_dh(a, alpha, d, theta) -> np.ndarray:
    """_summary_

    Args:
        a (_type_): _description_
        alpha (_type_): _description_
        d (_type_): _description_
        theta (_type_): _description_

    Returns:
        np.ndarray: Rotation Matrix
    """
    return np.array([
        [cos(theta),    -sin(theta)*cos(alpha),     sin(theta)*sin(alpha),  a*cos(theta)],
        [sin(theta),    cos(theta)*cos(alpha),      -cos(theta)*sin(alpha), a*sin(theta)],
        [0,             sin(alpha),                 cos(alpha),             d],
        [0,             0,                          0,                      1]
    ])

def translation(axis: Axis, amount):
    tmat = np.eye(4)
    tmat[axis.value,3] = amount
    return tmat

def rotation(axis: Axis, magnitude):
    rotmat = np.eye(4)
    if axis is Axis.X:
        rotmat[1,1] = cos(magnitude)
        rotmat[1,2] = -sin(magnitude)
        rotmat[2,1] = sin(magnitude)
        rotmat[2,2] = cos(magnitude)
    elif axis is Axis.Y:
        rotmat[0,0] = cos(magnitude)
        rotmat[0,2] = sin(magnitude)
        rotmat[2,0] = -sin(magnitude)
        rotmat[2,2] = cos(magnitude)
    else:  # Axis.Z
        rotmat[0,0] = cos(magnitude)
        rotmat[0,1] = -sin(magnitude)
        rotmat[1,0] = sin(magnitude)
        rotmat[1,1] = cos(magnitude)
    return rotmat

def get_htransform_mat(t: Transform, a: Axis, magnitude):
    if t is Transform.TRANSLATE:
        return translation(a, magnitude)
    else:
        return rotation(a, magnitude)


class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.joint_positions = [
            [0, 0, 0],   # 1
            [0, 0, 0],   # 2
            [0, 0, 0],   # 3
            [0, 0, 0],   # 4
            [0, 0, 0],   # 5
            [0, 0, 0],   # 6
            [0, 0, 0],   # 7
            [0, 0, 0],   # 8
        ]
        static_transforms = [
            [(TRANSLATE, Z, 0.141)],
            [(TRANSLATE, Z, 0.192), (ROT, X, -pi/2)],
            [(TRANSLATE, Y, -0.195), (ROT, X, pi/2)],
            [(TRANSLATE, X, 0.0825), (TRANSLATE, Z, 0.121), (ROT, X, pi/2)],
            [(TRANSLATE, Y, 0.125), (TRANSLATE, X, -0.0825), (ROT, X, -pi/2)],
            [(TRANSLATE, Z, 0.259), (TRANSLATE, Y, 0.015), (ROT, X, pi/2)],
            [(TRANSLATE, Z, 0.015), (TRANSLATE, Y, -0.051), (TRANSLATE, X, 0.088), (ROT, X, pi/2), (ROT, Z, -pi/4)],
            [(TRANSLATE, Z, 0.159)]
        ]
        self.precompositions = list()

        for transforms in static_transforms:
            tf = np.eye(4)
            for tf_spec in transforms:
                tf = tf@get_htransform_mat(*tf_spec)
            self.precompositions.append(tf)

    def get_transform_matrices(self, q):
        dh_params = [
            {'a':0.,        'alpha':0.,         'd':0.141,  'theta':q[0]},
            {'a':0.,        'alpha':-pi/2,      'd':0.192,  'theta':q[1]},
            {'a':0.,        'alpha':pi/2,       'd':0.,     'theta':q[2]},
            {'a':0.0825,    'alpha':-pi/2,      'd':0.316,  'theta':q[3]},
            {'a':0.,        'alpha':-pi/2,      'd':0.,     'theta':q[4]},
            {'a':0.384,     'alpha':0.,         'd':0.,     'theta':q[5]},
            {'a':0.,        'alpha':-pi/2,      'd':0.,     'theta':q[6]},
            {'a':0.088,     'alpha':0.,         'd':0.21,   'theta':pi/2},
        ]
        dynamic_transforms = [
            [(ROT, Z, q[0])],
            [(ROT, Z, q[1])],
            [(ROT, Z, q[2])],
            [(ROT, Z, q[3])],
            [(ROT, Z, q[4])],
            [(ROT, Z, q[5])],
            [(ROT, Z, q[6])],
            [(ROT, Z, 0)],
        ]
        dyn_tfs = list()
        for transforms in dynamic_transforms:
            tf = np.eye(4)
            for tf_spec in transforms:
                tf = tf@get_htransform_mat(*tf_spec)
            dyn_tfs.append(tf)
        rot_mats = [self.precompositions[i]@dyn_tf for i, dyn_tf in enumerate(dyn_tfs)]
        return rot_mats

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions - 8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        Tmat = np.identity(4)

        rot_mats = self.get_transform_matrices(q)

        saved_tf_mat = [None, None]
        idx_to_save = [4,5]

        for joint_index in range(8):
            Tmat = Tmat@(rot_mats[joint_index])
            if joint_index == idx_to_save[0]:
                saved_tf_mat[1] = Tmat
            elif joint_index == idx_to_save[1]:
                saved_tf_mat[0] = Tmat
            joint_pos = np.array(self.joint_positions[joint_index]+[1])
            jointPositions[joint_index] = (Tmat@joint_pos)[:3]

        # Your code ends here

        #return jointPositions, saved_tf_mat[0], saved_tf_mat[1]
        return jointPositions, Tmat

    # feel free to define additional helper methods to modularize your solution for lab 1


    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()


def tests():
    fk = FK()
    # Check end effector position
    l = np.array([ 0,    0,     0,     0,     0,    0,    0 ])
    jp, T0e = fk.forward(l)
    assert(np.isclose(jp[7], np.array([0.088, 0, 0.823])).all())

    # Check basic translation
    p = np.array([0, 0, 0, 1])
    mat = get_htransform_mat(TRANSLATE, Z, 1)
    p = mat@p
    assert(np.isclose(p, np.array([0,0,1,1])).all())

    # Check basic rotation
    p = np.array([0, 0, 1, 1])
    mat = get_htransform_mat(ROT, Z, 1)
    p = mat@p
    assert(np.isclose(p, np.array([0,0,1,1])).all())

    # Check active rotation
    p = np.array([0, 0, 1, 1])
    mat = get_htransform_mat(ROT, X, np.pi/2)
    p = mat@p
    assert(np.isclose(p, np.array([0,-1,0,1])).all())


if __name__ == "__main__":

    fk = FK()
    tests()

    # matches figure in the handout
    configurations = [
        np.array([ 0,    0,     0,     0,     0,    0,    0 ]),
        np.array([ 0,    pi/4,     0, -pi/4,     0, 0, 0 ]),
        np.array([ pi/2, 0,  pi/4, -pi/2, -pi/2, pi/2,    0 ]),
        np.array([ 0,    pi/4,     0, -pi/4,     0, 0, 2.89 ]),
        np.array([ -2.89, -1, 0,  -2.5, 0, 1,   1 ]),
    ]

    for configuration in configurations:
        joint_positions, T0e = fk.forward(configuration)

        print("Joint Positions:\n",joint_positions)
        print(joint_positions.shape)
        #print("End Effector Pose:\n",T0e)
