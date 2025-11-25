import numpy as np
from math import pi
from lib.calculateFK import FK

class IK:
    """
    Solves the 6 DOF (joint 5 fixed) IK problem for panda robot arm
    """
    limits = np.array([[-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973]])
    #  x offsets
    a1 = 0
    a2 = 0
    a3 = 0.0825
    a4 = 0.0825
    a5 = 0
    a6 = 0.088
    a7 = 0
    # z offset
    d1 = 0.333
    d2 = 0
    d3 = 0.316
    d4 = 0
    d5 = 0.384
    d6 = 0
    d7 = 0.210

    # This variable is used to express an arbitrary joint angle
    Q0 = 0.123



    def panda_ik(self, target, debug=False):
        """
        Solves 6 DOF IK problem given physical target in x, y, z space
        Args:
            target: dictionary containing:
                'R': numpy array of the end effector pose relative to the robot base
                't': numpy array of the end effector position relative to the robot base

        Returns:
             q = nx7 numpy array of joints in radians (q5: joint 5 angle should be 0)
        """
        R = target['R']
        # Student's code goes in between:

        # wristpos
        wrist = self.kin_decouple(target)
        # q4,6,7
        joints_467 = self.ik_pos(wrist)

        # Check for nan and delete
        nans = []
        size = len(joints_467)
        for i in range(size):
            if np.isnan(joints_467[i, 0]):
                nans.append(i)
        size = len(joints_467)

        joints_467 = np.delete(joints_467, nans, 0)

        # q1-3

        joints_123 = self.ik_orient(R, joints_467)

        q = np.zeros([joints_123.shape[0]*joints_467.shape[0], 7])

        print("467 Shape:", joints_467.shape)
        print("123 Shape:", joints_123.shape)
        print("Q Shape:", q.shape)

        for i in range(size):
            #get both sol
            for j in range(2):
                for x in range(3):
                    q[(i*2)+j, x] = joints_123[(i*2)+j, x]
                q[(i*2)+j, 3] = joints_467[i, 0]
                q[(i*2)+j, 4] = 0
                q[(i*2)+j, 5] = joints_467[i, 1]
                q[(i*2)+j, 6] = joints_467[i, 2]

        todelete = []
        for i in range(len(q)):
            for j in range(7):
                qindex=q[i,j]
                if (qindex < IK.limits[j, 0] or qindex > IK.limits[j, 1]):
                    todelete.append(i)

        q = np.delete(q, todelete, 0)

        # Student's code goes in between:

        ## DO NOT EDIT THIS PART
        # This will convert your joints output to the autograder format
        q = self.sort_joints(q)
        ## DO NOT EDIT THIS PART
        return q

    def kin_decouple(self, target):
        """
        Performs kinematic decoupling on the panda arm to find the position of wrist center
        Args:
            target: dictionary containing:
                'R': numpy array of the end effector pose relative to the robot base
                't': numpy array of the end effector position relative to the robot base

        Returns:
             wrist_pos = 3x1 numpy array of the position of the wrist center in frame 7
        """

        wrist_pos = np.zeros([3, 1])

        # Define variables and transposes
        R0_7 = target['R']
        t0_7 = target['t'].reshape(3,1)
        R7_0 = np.transpose(R0_7)
        t7_0 = -np.dot(R7_0, t0_7)
        d1 = IK.d1
        wristcol = np.array([[0], [0], [-1]])
        #  wrist_pos
        wrist_pos = t7_0 - d1*np.dot(R7_0, wristcol)

        return wrist_pos

    def ik_pos(self, wrist_pos):
        """
        Solves IK position problem on the joint 4, 6, 7
        Args:
            wrist_pos: 3x1 numpy array of the position of the wrist center in frame 7

        Returns:
             joints_467 = nx3 numpy array of all joint angles of joint 4, 6, 7
        """

        # Initialize joints_467
        joints_467 = np.zeros([4, 3])
        # infinite
        if wrist_pos[0] == 0 and wrist_pos[1] == 0:
            q7 = IK.Q0
        else:
            q7 = (((5*pi)/4) -np.arctan2(wrist_pos[1], wrist_pos[0]))[0]
        if q7 > pi:
            q7 = q7 - 2*pi
        if q7 > 0:
            q7prime = q7 - pi
        if q7 < 0:
            q7prime = q7 + pi
        if q7==0:
            q7prime = q7
        qusing=q7

        for l in range(2):
            if (l==0):
                qusing=q7
            if (l==1):
                qusing=q7prime
            for i in range(2):
                H7 = np.array([[np.cos(qusing-(pi/4)), -np.sin(qusing-(pi/4)), 0, 0],
                    [np.sin(qusing-(pi/4)), np.cos(qusing-(pi/4)), 0, 0],
                    [0, 0, 1, IK.d7],
                    [0, 0, 0, 1]])

                o7_2 = np.array([wrist_pos[0], wrist_pos[1], wrist_pos[2], [1]])
                o6_2 = (np.dot(H7, o7_2))[0:3]

                # Find dist
                d1 = ((IK.a4)**2 + (IK.d5)**2)**.5
                d2 = ((IK.a3)**2 + (IK.d3)**2)**.5


                costheta = ((o6_2[0]+IK.a6)**2 + (o6_2[2])**2 - (d1)**2 - (d2)**2)/(2*d1*d2)
                if costheta > 1 or costheta < -1:
                    q4 = np.nan
                    q6 = np.nan
                else:
                    theta2 = np.arccos(costheta)
                    if i == 1:
                        theta2 = -theta2
                    theta1 = np.arctan2(o6_2[2], (o6_2[0]+IK.a6)) - np.arctan2((d2*np.sin(theta2)), (d1 + d2*np.cos(theta2)))

                    q4 = (theta2 + np.arctan2(IK.d3, IK.a3) + np.arctan2(IK.d5, IK.a3) - pi)[0]
                    q6 = (theta1 - (pi/2) + np.arctan2(IK.a3, IK.d5))[0]

                    if q4 > pi:
                        q4 = q4 - 2*pi
                    if q4 < -pi:
                        q4 = q4 + 2*pi

                    if q6 > pi:
                        q6 = q6 - 2*pi
                    if q6 < -pi:
                        q6 = q6 + 2*pi
                # joints_467
                if(l==0):
                    joints_467[i, 0] = q4
                    joints_467[i, 1] = q6
                    joints_467[i, 2] = q7
                if(l==1):
                    joints_467[i+2, 0] = q4
                    joints_467[i+2, 1] = q6
                    joints_467[i+2, 2] = q7prime

        # Now, do same process but for second q7 value, if non-zero


        return joints_467

    def ik_orient(self, R, joints_467):
        """
        Solves IK orientation problem on the joint 1, 2, 3
        Args:
            R: numpy array of the end effector pose relative to the robot base
            joints_467: nx3 numpy array of all joint angles of joint 4, 6, 7

        Returns:
            joints_123 = nx3 numpy array of all joint angles of joint 1, 2 ,3
        """


        joints_123 = np.zeros([2*len(joints_467), 3])
        for i in range(len(joints_467)):
            q4 = joints_467[i, 0]
            q5 = 0
            q6 = joints_467[i, 1]
            q7 = joints_467[i, 2]
            A4 = np.array([[np.cos(q4), 0, -np.sin(q4), (IK.a4)*np.cos(q4)],
                [np.sin(q4), 0, np.cos(q4), (IK.a4)*np.sin(q4)],
                [0, -1, 0, 0],
                [0, 0, 0, 1]])
            A5 = np.array([[np.cos(q5), 0, np.sin(q5), 0],
                [np.sin(q5), 0, -np.cos(q5), 0],
                [0, 1, 0, IK.d5],
                [0, 0, 0, 1]])
            A6 = np.array([[np.cos(q6), 0, np.sin(q6), (IK.a6)*np.cos(q6)],
                [np.sin(q6), 0, -np.cos(q6), (IK.a6)*np.sin(q6)],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
            A7 = np.array([[np.cos(q7-(pi/4)), -np.sin(q7-(pi/4)), 0, 0],
                [np.sin(q7-(pi/4)), np.cos(q7-(pi/4)), 0, 0],
                [0, 0, 1, IK.d7],
                [0, 0, 0, 1]])
            A7_3 = np.transpose((A4@A5@A6@A7))
            R7_3 = A7_3[0:3, 0:3]
            R0_3 = np.dot(R, R7_3)

            q2 = np.arccos(R0_3[2, 1])
            if q2 == 0:
                q3 = IK.Q0
                q1 = IK.Q0
                joints_123[(i*2), : ] = [q1, q2, q3]
                joints_123[(i*2)+1, : ] = [1000, 1000, 1000]

            else:
                for j in range(2):
                    if j == 1:
                        q2 = -q2
                    cosq1 = R0_3[0, 1]/np.sin(q2)
                    sinq1 = R0_3[1, 1]/np.sin(q2)
                    q1 = np.arctan2(sinq1, cosq1)
                    cosq3 = -R0_3[2, 0]/np.sin(q2)
                    sinq3 = -R0_3[2, 2]/np.sin(q2)
                    q3 = np.arctan2(sinq3, cosq3)
                    if q7 == IK.Q0:
                        if j == 0:
                            joints_123[(i*2), : ] = [IK.Q0, q2, q3]
                            joints_123[(i*2)+1, : ] = [1000, 1000, 1000]
                    if q7 != IK.Q0:
                        joints_123[(i*2)+j, : ] = [q1, q2, q3]

        return joints_123

    def sort_joints(self, q, col=0):
        """
        Sort the joint angle matrix by ascending order
        Args:
            q: nx7 joint angle matrix
        Returns:
            q_as = nx7 joint angle matrix in ascending order
        """
        if col != 7:
            q_as = q[q[:, col].argsort()]
            for i in range(q_as.shape[0]-1):
                if (q_as[i, col] < q_as[i+1, col]):
                    # do nothing
                    pass
                else:
                    for j in range(i+1, q_as.shape[0]):
                        if q_as[i, col] < q_as[j, col]:
                            idx = j
                            break
                        elif j == q_as.shape[0]-1:
                            idx = q_as.shape[0]

                    q_as_part = self.sort_joints(q_as[i:idx, : ], col+1)
                    q_as[i:idx, : ] = q_as_part
        else:
            q_as = q[q[:, -1].argsort()]
        return q_as

def main():

    # fk solution code
    fk = FK()

    # input joints
    q1 = pi/2
    q2 = -1.76
    q3 = 0
    q4 = -3.07
    q6 = pi/2
    q7 = 0.78

    q_in  = np.array([q1, q2, q3, q4, 0, q6, q7])
    [_, T_fk] = fk.forward(q_in)

    # input of IK class
    target = {'R': T_fk[0:3, 0:3], 't': T_fk[0:3, 3]}
    ik = IK()
    q = ik.panda_ik(target)
    print('q value = ')
    print(q)

    # verify IK solutions
    for i in range(q.shape[0]):
        [_, T_ik] = fk.forward(q[i, :])
        print('Matrix difference = ')
        print(T_fk - T_ik)
        print()

if __name__ == '__main__':
    main()
