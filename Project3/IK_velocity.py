import numpy as np
from lib.calcJacobian import calcJacobian

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    J = calcJacobian(q_in)
    velocity = np.vstack((v_in, omega_in))

    # velocity = np.array([np.nan,1,2,3,np.nan,5])
    # J =  np.array([[0,1,2,3,4,5,6],
    #           [1,1,2,3,4,5,6],
    #           [2,1,2,3,4,5,6],
    #           [3,1,2,3,4,5,6],
    #           [4,1,2,3,4,5,6],
    #           [5,1,2,3,4,5,6]])

    v_total = []
    j_total = []
    count = 0
    
    for i in range(velocity.shape[0]):
        if np.isnan(velocity[i]):
            # print(i, velocity[i], " is nan")
            count+=1
        else:
            # print(i, velocity[i], " is not nan")
            v_total.append(velocity[i])
            j_total.append(J[i])
    if count == 6:
        return np.zeros((7, 1))

    v_total = np.array(v_total)
    j_total = np.array(j_total)

    # print('v_total', v_total)
    # print('j_total', j_total)

    dq = np.linalg.lstsq(j_total, v_total, rcond=None)[0]

    # print('dq', dq)
    return dq

# IK_velocity(0,0,0)