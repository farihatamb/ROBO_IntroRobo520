import numpy as np
from math import pi
from lib.calculateFK import FK

class IK:
    """
    Solves the 6 DOF (joint 5 fixed) IK problem for panda robot arm
    """


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
    
    limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]])


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
        print(joints_467)

        # q1-3

        joints_123 = self.ik_orient(R, joints_467)

        q_final = np.zeros([len(joints_123), 7])
        # Loop through size of initial matrix
        for i in range(len(joints_467)):
            #first iteration
            j1 = (i*2)
            q_final[j1, 0] = joints_123[j1, 0] #q1
            q_final[j1, 1] = joints_123[j1, 1] #q2
            q_final[j1, 2] = joints_123[j1, 2] #q3
            q_final[j1, 3] = joints_467[i, 0]     #q4
            q_final[j1, 4] = 0                    #q5
            q_final[j1, 5] = joints_467[i, 1]     #q6
            q_final[j1, 6] = joints_467[i, 2]     #q7
            ##second iteration for second solution
            j2 = (i*2)+1
            q_final[j2, 0] = joints_123[j2, 0] #q1
            q_final[j2, 1] = joints_123[j2, 1] #q2
            q_final[j2, 2] = joints_123[j2, 2] #q3
            q_final[j2, 3] = joints_467[i, 0]     #q4
            q_final[j2, 4] = 0                    #q5
            q_final[j2, 5] = joints_467[i, 1]     #q6
            q_final[j2, 6] = joints_467[i, 2]     #q7

        todelete = []
        for i in range(len(q_final)):
            remove=False
            for j in range(7):
                qindex=q_final[i,j]
                if (qindex < IK.limits[j, 0] or qindex > IK.limits[j, 1]):
                    if remove == False:
                        remove = True
                        todelete.append(i)

        q_final = np.delete(q_final, todelete, 0)

        # Student's code goes in between:

        ## DO NOT EDIT THIS PART
        # This will convert your joints output to the autograder format
        q = self.sort_joints(q_final)
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
        d1 = IK.d1


        # Define variables and transposes
        rot_07 = target['R']
        t_07 = target['t'].reshape(3,1)
        rot_70 = np.transpose(rot_07)
        t_70 = -np.dot(rot_70, t_07)
        wristcol = np.array([[0], [0], [-1]])
        #  wrist_pos
        wrist_pos = t_70 - d1*np.dot(rot_70, wristcol)

        return wrist_pos

    def ik_pos(self, wrist_pos):
        """
        Solves IK position problem on the joint 4, 6, 7
        Args:
            wrist_pos: 3x1 numpy array of the position of the wrist center in frame 7

        Returns:
             joints_467 = nx3 numpy array of all joint angles of joint 4, 6, 7
        """

        # Initialize joints_467 array

        joints_467 = np.zeros([4, 3])

        d1 = IK.d1
        d2 = IK.d2
        d3 = IK.d3
        d4 = IK.d4
        d5 = IK.d5
        d6 = IK.d6
        d7 = IK.d7

        a1 = IK.a1
        a2 = IK.a2
        a3 = IK.a3
        a4 = IK.a4
        a5 = IK.a5
        a6 = IK.a6
        a7 = IK.a7

        Q0 = IK.Q0

        wrist_pos = np.array([wrist_pos[0,0], wrist_pos[1,0], wrist_pos[2,0]])
        # checking for infinite solutions cases
        if wrist_pos[0] == 0 and wrist_pos[1] == 0:
            q7 = Q0		# if both x and y coords of wrist pos are equal to 0, q7 has infinite solutions
        else:			# if infinite solutions case passed enter this statement
            q7 = (5*pi)/4 - np.arctan2(wrist_pos[1], wrist_pos[0])	# finding q7 using arctan

        # check for q7 values above pi:
        if q7 > pi:
            q7 = q7 - 2*pi

        # check for every case of q7 possible:
        if q7 > 0:
            q7prime = q7 - pi
        elif q7 < 0:
            q7prime = q7 + pi
        elif q7==0:
            q7prime = 0


        # each q7 generates two sets of q4 and q6 --> 4 angle sets
        # starting with the original q7 we found:

        for case in range(2):
            # this matrix is used to find the transformation from 7 to 2
            H27prime = np.array([[np.cos(q7-(pi/4)), -np.sin(q7-(pi/4)), 0, 0],
                                 [np.sin(q7-(pi/4)), np.cos(q7-(pi/4)), 0, 0],
                                 [0, 0, 1, d7],
                                 [0, 0, 0, 1]])

            # can use this matrix to find the distance between o6 and o2
            o72 = np.array([wrist_pos[0], wrist_pos[1], wrist_pos[2], 1])
            o62 = H27prime@o72
            o62 = o62[0:3].flatten()
            a6_offset = np.array([a6, 0, 0])
            o62 = o62 + a6_offset
            o62 = o62.flatten()

            # now we will be deconstructing the geometry of this arm:
            hyp1 = (d5**2 + a4**2)**0.5
            hyp2 = (d3**2 + a3**2)**0.5

            # find theta 2:
            l_ = np.sqrt(o62[0]**2 + o62[2]**2)
            costheta2 = (l_**2 - (hyp1**2 + hyp2**2))/(2*hyp1*hyp2)

            # if costheta2 is any value greater than 1 or less than -1
            if costheta2 > 1 or costheta2 < -1:
                q4 = np.nan
                q6 = np.nan
            else:
                theta2 = np.arccos(costheta2)
                if case == 0:           #flip sign for elbow up/elbow down
                    theta2 = theta2
                elif case == 1:
                    theta2 = -theta2

            sin_of_theta2 = np.sin(theta2)
            cos_of_theta2 = np.cos(theta2)

            # calculate theta 1 using the slides from lecture 17
            theta1 = np.arctan2(o62[2], o62[0]) - np.arctan2((hyp2*sin_of_theta2),
                (hyp1 + hyp2*np.cos(theta2)))

            # compute q4 and q6 for this set
            q4 = theta2 + np.arctan2(IK.d3, IK.a3) + np.arctan2(IK.d5, IK.a4) - pi
            q6 = theta1 + np.arctan2(IK.a3, IK.d5) - pi/2

            # Checks for boundary exceptions
            if q4 > pi:
                q4 = q4 - 2*pi
            elif q4 < -pi:
                q4 = q4 + 2*pi

            if q6 > pi:
                q6 = q6 - 2*pi
            elif q6 < -pi:
                q6 = q6 + 2*pi

            # Populates joints_467
            joints_467[case, 0] = q4
            joints_467[case, 1] = q6
            joints_467[case, 2] = q7

        # this matrix is used to find the
        for case2 in range(2):
            H27prime2 = np.array([[np.cos(q7prime-(pi/4)), -np.sin(q7prime-(pi/4)), 0, 0],
                                 [np.sin(q7prime-(pi/4)), np.cos(q7prime-(pi/4)), 0, 0],
                                 [0, 0, 1, IK.d7],
                                 [0, 0, 0, 1]])

            # can use this matrix to find the distance between o6 and o2
            o72 = np.array([wrist_pos[0], wrist_pos[1], wrist_pos[2], 1])
            o62 = H27prime2@o72
            o62 = o62[0:3].flatten()
            a6_offset = np.array([IK.a6, 0, 0])
            o62 = o62 + a6_offset

            # now we will be deconstructing the geometry of this arm:
            hyp1 = (IK.d5**2 + IK.a4**2)**0.5
            hyp2 = (IK.d3**2 + IK.a3**2)**0.5

            # find theta 2:
            l_ = np.linalg.norm(o62)
            costheta2 = (l_**2 - (hyp1**2 + hyp2**2))/(2*hyp1*hyp2)

            # if costheta2 is any value greater than 1 or less than -1
            if costheta2 > 1 or costheta2 < -1:
                q4 = np.nan
                q6 = np.nan
            else:
                theta2 = np.arccos(costheta2)
                if case2 == 0:           #flip sign for elbow up/elbow down
                    theta2 = theta2
                elif case2 == 1:
                    theta2 = -theta2

                sin_of_theta2 = np.sin(theta2)
                cos_of_theta2 = np.cos(theta2)

                # calculate theta 1 using the slides from lecture 17
                theta1 = np.arctan2(o62[2], o62[0]) - np.arctan2((hyp2*sin_of_theta2),
                    (hyp1 + hyp2*np.cos(theta2)))

                # compute q4 and q6 for this set
                q4 = theta2 + np.arctan2(IK.d3, IK.a3) + np.arctan2(IK.d5, IK.a4) - pi
                q6 = theta1 + np.arctan2(IK.a3, IK.d5) - pi/2

                # Checks for boundary exceptions
                if q4 > pi:
                    q4 = q4 - 2*pi
                elif q4 < -pi:
                    q4 = q4 + 2*pi

                if q6 > pi:
                    q6 = q6 - 2*pi
                elif q6 < -pi:
                    q6 = q6 + 2*pi

            # Populates joints_467
            joints_467[2+case2, 0] = q4
            joints_467[2+case2, 1] = q6
            joints_467[2+case2, 2] = q7

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
        d1 = IK.d1
        d2 = IK.d2
        d3 = IK.d3
        d4 = IK.d4
        d5 = IK.d5
        d6 = IK.d6
        d7 = IK.d7

        a1 = IK.a1
        a2 = IK.a2
        a3 = IK.a3
        a4 = IK.a4
        a5 = IK.a5
        a6 = IK.a6
        a7 = IK.a7

        Q0 = IK.Q0

        joints_123 = np.zeros([2*len(joints_467), 3])
        for i in range(len(joints_467)):
            q4 = joints_467[i, 0]
            q5 = 0
            q6 = joints_467[i, 1]
            q7 = joints_467[i, 2]
            T4 = np.array([[np.cos(q4), 0, -np.sin(q4), (a4)*np.cos(q4)],
                [np.sin(q4), 0, np.cos(q4), (a4)*np.sin(q4)],
                [0, -1, 0, 0],
                [0, 0, 0, 1]])
            T5 = np.array([[np.cos(q5), 0, np.sin(q5), 0],
                [np.sin(q5), 0, -np.cos(q5), 0],
                [0, 1, 0, d5],
                [0, 0, 0, 1]])
            T6 = np.array([[np.cos(q6), 0, np.sin(q6), (a6)*np.cos(q6)],
                [np.sin(q6), 0, -np.cos(q6), (a6)*np.sin(q6)],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
            T7 = np.array([[np.cos(q7-(pi/4)), -np.sin(q7-(pi/4)), 0, 0],
                [np.sin(q7-(pi/4)), np.cos(q7-(pi/4)), 0, 0],
                [0, 0, 1, d7],
                [0, 0, 0, 1]])
            T7_3 = np.transpose((T4@T5@T6@T7))
            R7_3 = T7_3[0:3, 0:3]
            R0_3 = np.dot(R, R7_3)

            q2 = np.arccos(R0_3[2, 1])
            if q2 == 0:
                q3 = Q0
                q1 = Q0
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
                    if q7 == Q0:
                        if j == 0:
                            joints_123[(i*2), : ] = [Q0, q2, q3]
                            joints_123[(i*2)+1, : ] = [1000, 1000, 1000]
                    if q7 != Q0:
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
