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

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    v = np.concatenate([v_in, omega_in])

    J = calcJacobian(q_in)
    valid_idxs = list()
    for i in range(6):
        if not np.isnan(v[i]):
            valid_idxs.append(i)
    #J = J[valid_idxs]
    #v = v[valid_idxs]
    if (len(valid_idxs)==0):
        return np.array([0,0,0,0,0,0,0])
    dq, res, rank, s = np.linalg.lstsq(J, v, rcond=None)
    dq=dq.reshape(7,)
    return dq


if __name__ == "__main__":
    q_in = np.zeros((7))
    v_in = np.array([0, 0, 1])
    w_in = np.array([0, 0, 0])
    IKV = IK_velocity(q_in, v_in, w_in)
    print(IKV)