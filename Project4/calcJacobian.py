import numpy as np
from lib.calculateFK import FK

def compute_T(theta, alpha, d, a):
    mat = np.array([[ np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                    [ np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                    [ 0,              np.sin(alpha),                np.cos(alpha),              d],
                    [ 0,              0,             0, 1]])
    return mat

def skew(w):
  res = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]])

  return res

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
	#x direction offsets
    x1 = 0
    x2 = 0
    x3 = 0.0825
    x4 = -0.0825
    x5 = 0
    x6 = 0.088
    x7 = 0

    #z direction offsets
    z0 = .141 #world to base offset
    z1 = 0.333
    z2 = 0
    z3 = 0.316
    z4 = 0
    z5 = 0.384
    z6 = 0
    z7 = 0.210

    #joint angles
    q1 = q_in[0]
    q2 = q_in[1]
    q3 = q_in[2]
    q4 = q_in[3]
    q5 = q_in[4]
    q6 = q_in[5]
    q7 = q_in[6] - np.pi/4



    params = np.array([
    [0, -np.pi/2, z1, q1],
    [0, np.pi/2, 0, q2],
    [x3, np.pi/2, z3, q3],
    [x4, -np.pi/2, 0, q4],
    [0, np.pi/2, z5, q5],
    [x6, np.pi/2, 0, q6],
    [0, 0, z7, q7]])
    ##compute_A(theta, alpha, d, a)

    T01 = compute_T(params[0][0], params[0][1], params[0][2], params[0][3])
    T12 = compute_T(params[1][0], params[1][1], params[1][2], params[1][3])
    T23 = compute_T(params[2][0], params[2][1], params[2][2], params[2][3])
    T34 = compute_T(params[3][0], params[3][1], params[3][2], params[3][3])
    T45 = compute_T(params[4][0], params[4][1], params[4][2], params[4][3])
    T56 = compute_T(params[5][0], params[5][1], params[5][2], params[5][3])
    T67 = compute_T(params[6][0], params[6][1], params[6][2], params[6][3])


    T00 = np.array([	[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, .141],
			[0, 0, 0, 1]	])
    T01 = T01
    T02 = T01 @ T12
    T03 = T01 @ T12 @ T23
    T04 = T01 @ T12 @ T23 @ T34
    T05 = T01 @ T12 @ T23 @ T34 @ T45
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    T07 = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67

    p0e = T07[:3,3]

    ##Linear Velocities
    #lv1 = skew(T01[:,2][:-1]) @ (p0e - T01[:,3][:-1]) #np.cross??
    #lv2 = skew(T02[:,2][:-1]) @ (p0e - T02[:,3][:-1])
    #lv3 = skew(T03[:,2][:-1]) @ (p0e - T03[:,3][:-1])
    #lv4 = skew(T04[:,2][:-1]) @ (p0e - T04[:,3][:-1])
    #lv5 = skew(T05[:,2][:-1]) @ (p0e - T05[:,3][:-1])
    #lv6 = skew(T06[:,2][:-1]) @ (p0e - T06[:,3][:-1])
    #lv7 = skew(T07[:,2][:-1]) @ (p0e - T07[:,3][:-1])

    lv1 = np.cross(T00[:3,2], (p0e - T00[:3,3]))
    lv2 = np.cross(T01[:3,2], (p0e - T01[:3,3]))
    lv3 = np.cross(T02[:3,2], (p0e - T02[:3,3]))
    lv4 = np.cross(T03[:3,2], (p0e - T03[:3,3]))
    lv5 = np.cross(T04[:3,2], (p0e - T04[:3,3]))
    lv6 = np.cross(T05[:3,2], (p0e - T05[:3,3]))
    lv7 = np.cross(T06[:3,2], (p0e - T06[:3,3]))

    ##Angular Velocities
    av1 = T00[:,2][:-1]
    av2 = T01[:,2][:-1]
    av3 = T02[:,2][:-1]
    av4 = T03[:,2][:-1]
    av5 = T04[:,2][:-1]
    av6 = T05[:,2][:-1]
    av7 = T06[:,2][:-1]


    J = np.array([       [lv1[0], lv1[1], lv1[2], av1[0], av1[1], av1[2]],
                         [lv2[0], lv2[1], lv2[2], av2[0], av2[1], av2[2]],
                         [lv3[0], lv3[1], lv3[2], av3[0], av3[1], av3[2]],
                         [lv4[0], lv4[1], lv4[2], av4[0], av4[1], av4[2]],
                         [lv5[0], lv5[1], lv5[2], av5[0], av5[1], av5[2]],
                         [lv6[0], lv6[1], lv6[2], av6[0], av6[1], av6[2]],
                         [lv7[0], lv7[1], lv7[2], av7[0], av7[1], av7[2]]
                  ])

    return J.T

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))

