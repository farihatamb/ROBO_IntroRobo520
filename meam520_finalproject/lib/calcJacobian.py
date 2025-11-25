import numpy as np
from numpy import sin, cos, sqrt
#from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## Constants
    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = q_in
    d_1 = 0.333
    d_3 = 0.316
    a_3 = a_4 = 0.0825
    d_5 = 0.384
    a_6 = 0.088
    d_7 = 0.210
    ## Linear Jacobian
    J[0, 0] = -a_3*sin(q_1)*cos(q_2)*cos(q_3) - a_3*sin(q_3)*cos(q_1) - a_4*(-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*cos(q_4) + a_4*sin(q_1)*sin(q_2)*sin(q_4) + a_6*(((-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*cos(q_4) - sin(q_1)*sin(q_2)*sin(q_4))*cos(q_5) + (sin(q_1)*sin(q_3)*cos(q_2) - cos(q_1)*cos(q_3))*sin(q_5))*cos(q_6) + a_6*(-(-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_4) - sin(q_1)*sin(q_2)*cos(q_4))*sin(q_6) - d_3*sin(q_1)*sin(q_2) + d_5*(-(-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_4) - sin(q_1)*sin(q_2)*cos(q_4)) + d_7*((((-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*cos(q_4) - sin(q_1)*sin(q_2)*sin(q_4))*cos(q_5) + (sin(q_1)*sin(q_3)*cos(q_2) - cos(q_1)*cos(q_3))*sin(q_5))*sin(q_6) - (-(-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_4) - sin(q_1)*sin(q_2)*cos(q_4))*cos(q_6))
    J[1, 0] = -a_3*sin(q_1)*sin(q_3) + a_3*cos(q_1)*cos(q_2)*cos(q_3) - a_4*(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) - a_4*sin(q_2)*sin(q_4)*cos(q_1) + a_6*(((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*cos(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_5))*cos(q_6) + a_6*(-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*sin(q_6) + d_3*sin(q_2)*cos(q_1) + d_5*(-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4)) + d_7*((((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*cos(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_5))*sin(q_6) - (-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*cos(q_6))
    J[2, 0] = 0

    J[0, 1] = -a_3*sin(q_2)*cos(q_1)*cos(q_3) + a_4*sin(q_2)*cos(q_1)*cos(q_3)*cos(q_4) - a_4*sin(q_4)*cos(q_1)*cos(q_2) + a_6*((-sin(q_2)*cos(q_1)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_1)*cos(q_2))*cos(q_5) + sin(q_2)*sin(q_3)*sin(q_5)*cos(q_1))*cos(q_6) + a_6*(sin(q_2)*sin(q_4)*cos(q_1)*cos(q_3) + cos(q_1)*cos(q_2)*cos(q_4))*sin(q_6) + d_3*cos(q_1)*cos(q_2) + d_5*(sin(q_2)*sin(q_4)*cos(q_1)*cos(q_3) + cos(q_1)*cos(q_2)*cos(q_4)) + d_7*(((-sin(q_2)*cos(q_1)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_1)*cos(q_2))*cos(q_5) + sin(q_2)*sin(q_3)*sin(q_5)*cos(q_1))*sin(q_6) - (sin(q_2)*sin(q_4)*cos(q_1)*cos(q_3) + cos(q_1)*cos(q_2)*cos(q_4))*cos(q_6))
    J[1, 1] = -a_3*sin(q_1)*sin(q_2)*cos(q_3) + a_4*sin(q_1)*sin(q_2)*cos(q_3)*cos(q_4) - a_4*sin(q_1)*sin(q_4)*cos(q_2) + a_6*((-sin(q_1)*sin(q_2)*cos(q_3)*cos(q_4) + sin(q_1)*sin(q_4)*cos(q_2))*cos(q_5) + sin(q_1)*sin(q_2)*sin(q_3)*sin(q_5))*cos(q_6) + a_6*(sin(q_1)*sin(q_2)*sin(q_4)*cos(q_3) + sin(q_1)*cos(q_2)*cos(q_4))*sin(q_6) + d_3*sin(q_1)*cos(q_2) + d_5*(sin(q_1)*sin(q_2)*sin(q_4)*cos(q_3) + sin(q_1)*cos(q_2)*cos(q_4)) + d_7*(((-sin(q_1)*sin(q_2)*cos(q_3)*cos(q_4) + sin(q_1)*sin(q_4)*cos(q_2))*cos(q_5) + sin(q_1)*sin(q_2)*sin(q_3)*sin(q_5))*sin(q_6) - (sin(q_1)*sin(q_2)*sin(q_4)*cos(q_3) + sin(q_1)*cos(q_2)*cos(q_4))*cos(q_6))
    J[2, 1] = -a_3*cos(q_2)*cos(q_3) + a_4*sin(q_2)*sin(q_4) + a_4*cos(q_2)*cos(q_3)*cos(q_4) + a_6*((-sin(q_2)*sin(q_4) - cos(q_2)*cos(q_3)*cos(q_4))*cos(q_5) + sin(q_3)*sin(q_5)*cos(q_2))*cos(q_6) + a_6*(-sin(q_2)*cos(q_4) + sin(q_4)*cos(q_2)*cos(q_3))*sin(q_6) - d_3*sin(q_2) + d_5*(-sin(q_2)*cos(q_4) + sin(q_4)*cos(q_2)*cos(q_3)) + d_7*(((-sin(q_2)*sin(q_4) - cos(q_2)*cos(q_3)*cos(q_4))*cos(q_5) + sin(q_3)*sin(q_5)*cos(q_2))*sin(q_6) - (-sin(q_2)*cos(q_4) + sin(q_4)*cos(q_2)*cos(q_3))*cos(q_6))

    J[0, 2] = -a_3*sin(q_1)*cos(q_3) - a_3*sin(q_3)*cos(q_1)*cos(q_2) - a_4*(-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*cos(q_4) + a_6*((sin(q_1)*sin(q_3) - cos(q_1)*cos(q_2)*cos(q_3))*sin(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*cos(q_4)*cos(q_5))*cos(q_6) - a_6*(-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_4)*sin(q_6) - d_5*(-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_4) + d_7*(((sin(q_1)*sin(q_3) - cos(q_1)*cos(q_2)*cos(q_3))*sin(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*cos(q_4)*cos(q_5))*sin(q_6) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_4)*cos(q_6))
    J[1, 2] = -a_3*sin(q_1)*sin(q_3)*cos(q_2) + a_3*cos(q_1)*cos(q_3) - a_4*(-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*cos(q_4) + a_6*((-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*cos(q_4)*cos(q_5) + (-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_5))*cos(q_6) - a_6*(-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*sin(q_4)*sin(q_6) - d_5*(-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*sin(q_4) + d_7*(((-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*cos(q_4)*cos(q_5) + (-sin(q_1)*cos(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_5))*sin(q_6) + (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*sin(q_4)*cos(q_6))
    J[2, 2] = a_3*sin(q_2)*sin(q_3) - a_4*sin(q_2)*sin(q_3)*cos(q_4) + a_6*(sin(q_2)*sin(q_3)*cos(q_4)*cos(q_5) + sin(q_2)*sin(q_5)*cos(q_3))*cos(q_6) - a_6*sin(q_2)*sin(q_3)*sin(q_4)*sin(q_6) - d_5*sin(q_2)*sin(q_3)*sin(q_4) + d_7*((sin(q_2)*sin(q_3)*cos(q_4)*cos(q_5) + sin(q_2)*sin(q_5)*cos(q_3))*sin(q_6) + sin(q_2)*sin(q_3)*sin(q_4)*cos(q_6))

    J[0, 3] = a_4*(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) - a_4*sin(q_2)*cos(q_1)*cos(q_4) + a_6*(-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*cos(q_5)*cos(q_6) + a_6*(-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) - sin(q_2)*sin(q_4)*cos(q_1))*sin(q_6) + d_5*(-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) - sin(q_2)*sin(q_4)*cos(q_1)) + d_7*((-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*sin(q_6)*cos(q_5) - (-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) - sin(q_2)*sin(q_4)*cos(q_1))*cos(q_6))
    J[1, 3] = a_4*(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) - a_4*sin(q_1)*sin(q_2)*cos(q_4) + a_6*(-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*sin(q_2)*cos(q_4))*cos(q_5)*cos(q_6) + a_6*(-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) - sin(q_1)*sin(q_2)*sin(q_4))*sin(q_6) + d_5*(-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) - sin(q_1)*sin(q_2)*sin(q_4)) + d_7*((-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*sin(q_2)*cos(q_4))*sin(q_6)*cos(q_5) - (-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) - sin(q_1)*sin(q_2)*sin(q_4))*cos(q_6))
    J[2, 3] = -a_4*sin(q_2)*sin(q_4)*cos(q_3) - a_4*cos(q_2)*cos(q_4) + a_6*(sin(q_2)*sin(q_4)*cos(q_3) + cos(q_2)*cos(q_4))*cos(q_5)*cos(q_6) + a_6*(sin(q_2)*cos(q_3)*cos(q_4) - sin(q_4)*cos(q_2))*sin(q_6) + d_5*(sin(q_2)*cos(q_3)*cos(q_4) - sin(q_4)*cos(q_2)) + d_7*((sin(q_2)*sin(q_4)*cos(q_3) + cos(q_2)*cos(q_4))*sin(q_6)*cos(q_5) - (sin(q_2)*cos(q_3)*cos(q_4) - sin(q_4)*cos(q_2))*cos(q_6))

    J[0, 4] = a_6*(-((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*sin(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*cos(q_5))*cos(q_6) + d_7*(-((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*sin(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*cos(q_5))*sin(q_6)
    J[1, 4] = a_6*(-((sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_2)*sin(q_4))*sin(q_5) + (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*cos(q_5))*cos(q_6) + d_7*(-((sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_2)*sin(q_4))*sin(q_5) + (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*cos(q_5))*sin(q_6)
    J[2, 4] = a_6*(-(-sin(q_2)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_2))*sin(q_5) + sin(q_2)*sin(q_3)*cos(q_5))*cos(q_6) + d_7*(-(-sin(q_2)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_2))*sin(q_5) + sin(q_2)*sin(q_3)*cos(q_5))*sin(q_6)

    J[0, 5] = -a_6*(((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*cos(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_5))*sin(q_6) + a_6*(-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*cos(q_6) + d_7*((((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*cos(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_5))*cos(q_6) + (-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*sin(q_6))
    J[1, 5] = -a_6*(((sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_2)*sin(q_4))*cos(q_5) + (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*sin(q_5))*sin(q_6) + a_6*(-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*sin(q_2)*cos(q_4))*cos(q_6) + d_7*((((sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_2)*sin(q_4))*cos(q_5) + (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*sin(q_5))*cos(q_6) + (-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*sin(q_2)*cos(q_4))*sin(q_6))
    J[2, 5] = -a_6*((-sin(q_2)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_2))*cos(q_5) + sin(q_2)*sin(q_3)*sin(q_5))*sin(q_6) + a_6*(sin(q_2)*sin(q_4)*cos(q_3) + cos(q_2)*cos(q_4))*cos(q_6) + d_7*(((-sin(q_2)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_2))*cos(q_5) + sin(q_2)*sin(q_3)*sin(q_5))*cos(q_6) + (sin(q_2)*sin(q_4)*cos(q_3) + cos(q_2)*cos(q_4))*sin(q_6))

    J[0, 6] = 0
    J[1, 6] = 0
    J[2, 6] = 0


    ## Rotational Jacobian

    J[3, 0] = (0)
    J[4, 0] = (0)
    J[5, 0] = (1)

    J[3, 1] = (-sin(q_1))
    J[4, 1] = (cos(q_1))
    J[5, 1] = (0)

    J[3, 2] = (sin(q_2)*cos(q_1))
    J[4, 2] = (sin(q_1)*sin(q_2))
    J[5, 2] = (cos(q_2))

    J[3, 3] = (sin(q_1)*cos(q_3) + sin(q_3)*cos(q_1)*cos(q_2))
    J[4, 3] = (sin(q_1)*sin(q_3)*cos(q_2) - cos(q_1)*cos(q_3))
    J[5, 3] = (-sin(q_2)*sin(q_3))

    J[3, 4] = (-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))
    J[4, 4] = (-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*sin(q_2)*cos(q_4))
    J[5, 4] = (sin(q_2)*sin(q_4)*cos(q_3) + cos(q_2)*cos(q_4))

    J[3, 5] = (((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*sin(q_5) - (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*cos(q_5))
    J[4, 5] = (((sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_2)*sin(q_4))*sin(q_5) - (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*cos(q_5))
    J[5, 5] = ((-sin(q_2)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_2))*sin(q_5) - sin(q_2)*sin(q_3)*cos(q_5))

    J[3, 6] = ((((-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*cos(q_4) + sin(q_2)*sin(q_4)*cos(q_1))*cos(q_5) + (-sin(q_1)*cos(q_3) - sin(q_3)*cos(q_1)*cos(q_2))*sin(q_5))*sin(q_6) - (-(-sin(q_1)*sin(q_3) + cos(q_1)*cos(q_2)*cos(q_3))*sin(q_4) + sin(q_2)*cos(q_1)*cos(q_4))*cos(q_6))
    J[4, 6] = ((((sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_2)*sin(q_4))*cos(q_5) + (-sin(q_1)*sin(q_3)*cos(q_2) + cos(q_1)*cos(q_3))*sin(q_5))*sin(q_6) - (-(sin(q_1)*cos(q_2)*cos(q_3) + sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*sin(q_2)*cos(q_4))*cos(q_6))
    J[5, 6] = (((-sin(q_2)*cos(q_3)*cos(q_4) + sin(q_4)*cos(q_2))*cos(q_5) + sin(q_2)*sin(q_3)*sin(q_5))*sin(q_6) - (sin(q_2)*sin(q_4)*cos(q_3) + cos(q_2)*cos(q_4))*cos(q_6))

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
