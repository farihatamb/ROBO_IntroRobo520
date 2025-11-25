import numpy as np
import copy
from math import pi
import pprint
from lib.calculateFK import FK, matrix_from_dh

class IK:
    """
    Solves the 6 DOF (joint 5 fixed) IK problem for panda robot arm
    """
    # offsets along x direction 
    a1 = 0 
    a2 = 0
    a3 = 0.0825
    a4 = 0.0825
    a5 = 0 
    a6 = 0.088
    a7 = 0

    # offsets along z direction 
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
        J1, J2, J3, J4, J5, J6, J7 = 0, 1, 2, 3, 4, 5, 6

        if not debug:
            wrist_pos = self.kin_decouple(target)
            joints_467 = self.ik_pos(wrist_pos, target)
            joints_123 = self.ik_orient(target['R'], joints_467)
        else:    
            wrist_pos = self.kin_decouple(target)
            joints_467, pack_467 = self.ik_pos(wrist_pos, target, debug=debug)
            joints_123, pack_123 = self.ik_orient(target['R'], joints_467, debug=debug)
        q = np.zeros((4,7))
        q[:,:3] = joints_123
        q[:,J4] = joints_467[:,0]
        q[:,J6] = joints_467[:,1]
        q[:,J7] = joints_467[:,2]
        
        fk = FK()

        # TODO: Filtering out invalid solutions


        ## DO NOT EDIT THIS PART 
        # This will convert your joints output to the autograder format
        #q = self.sort_joints(q)
        ## DO NOT EDIT THIS PART
        if not debug:
            return np.zeros((0,3))
        else:
            return q, [pack_467[0]['f6'], pack_467[0]['f5'], pack_467[0]['f4'], pack_467[0]['f3']]

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
        wrist_pos = (-target['R'].T)@target['t'] + self.d1*(target['R'].T@np.array([0, 0, 1]))
        return wrist_pos 

    def ik_pos(self, wrist_pos, target, debug=False):
        """
        Solves IK position problem on the joint 4, 6, 7 
        Args: 
            wrist_pos: 3x1 numpy array of the position of the wrist center in frame 7

        Returns:
             joints_467 = nx3 numpy array of all joint angles of joint 4, 6, 7
        """
        J4, J6, J7 = 0, 1, 2
        joint_vals = list()
        # Joint 7 is the orientation joint, which is calculated using the 7,2 vector
        joint_vals.append({
            "j7": np.float64((np.arctan2(-wrist_pos[1], wrist_pos[0]) + np.pi/4)),
            "wrist_pos": wrist_pos
        })
        joint_vals.append({
            "j7": np.float64((np.arctan2(-wrist_pos[1], wrist_pos[0]) + np.pi/4) + np.pi),
            "wrist_pos": wrist_pos
        })
        for idx, pack in enumerate(joint_vals):
            # Ensure we lie within joint limits
            if pack['j7'] > np.pi:
                pack['j7'] -= 2*np.pi
            elif pack['j7'] < -np.pi:
                pack['j7'] += 2*np.pi
            # Get the matrix that transforms o6 to o7
            joint_vals[idx]["t_mat"] = matrix_from_dh(0, 0, self.d7, pack['j7']-np.pi/4)
            # We can solve for q4 and q6 as if they are planar RR
            # o^6_{2-5} = T^6_7(q7)@o^7_2 - [-a6, 0, 0]
            # Compute the location of o6 by using the transpose of the rotation matrix
            joint_vals[idx]['o625'] = joint_vals[idx]["t_mat"][:3, :3].T@wrist_pos - np.array([-self.a6, 0, 0])
            joint_vals[idx]['l'] = np.sqrt(joint_vals[idx]["o625"][0]**2 + joint_vals[idx]["o625"][2]**2)
            joint_vals[idx]['a1'] = np.sqrt(self.a3**2 + self.d5**2)
            joint_vals[idx]['a2'] = np.sqrt(self.a3**2 + self.d3**2)
        
        elbow_up = copy.deepcopy(joint_vals)
        elbow_down = copy.deepcopy(joint_vals)
        for idx, pack in enumerate(elbow_up):
            pack["theta_2"] = np.arccos((pack['l']**2 - pack['a1']**2 - pack['a2']**2)/(2*pack['a1']*pack['a2']))
        for pack in elbow_down:
            pack["theta_2"] = -np.arccos((pack['l']**2 - pack['a1']**2 - pack['a2']**2)/(2*pack['a1']*pack['a2']))
        
        joint_vals = elbow_down + elbow_up

        for pack in joint_vals:
            pack['theta_1'] = np.arctan2(pack['o625'][2], pack['o625'][0]) - np.arctan2(pack['a2']*np.sin(pack['theta_2']), pack['a1']+pack['a2']*np.cos(pack['theta_2']))
            pack['j6'] = pack['theta_1'] + np.arctan2(self.a3,self.d5) - np.pi/2
            pack['j4'] = np.pi - pack['theta_2'] - np.arctan2(self.d3, self.a3) - np.arctan2(self.d5, self.a3)
        
        joints_467 = np.zeros((len(joint_vals),3))
        for i, pack in enumerate(joint_vals):
            joints_467[i] = np.array([pack['j4'],pack['j6'],pack['j7']])

        if not debug:
            return joints_467
        else:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(joint_vals)
            for pack in joint_vals:
                # Calculate the frame transformations:
                end_effector_frame = np.vstack((np.hstack((target['R'], np.array([target['t']]).T)), np.array([[0, 0, 0, 1]])))
                pack['f6'] = (end_effector_frame@np.linalg.inv(matrix_from_dh(0, 0, self.d7, pack['j7']-(np.pi/4))), "frame6")
                pack['f5'] = (pack['f6'][0]@np.linalg.inv(matrix_from_dh(self.a6, np.pi/2, 0, pack['j6'])) , "frame5")
                pack['f4'] = (pack['f5'][0]@np.linalg.inv(matrix_from_dh(0, np.pi/2, self.d5, 0)) , "frame4")
                pack['f3'] = (pack['f4'][0]@np.linalg.inv(matrix_from_dh(0, np.pi/2, self.d5, 0)), "frame3")
            return joints_467, joint_vals

    def ik_orient(self, R, joints_467, debug=False):
        """
        Solves IK orientation problem on the joint 1, 2, 3
        Args: 
            R: numpy array of the end effector pose relative to the robot base 
            joints_467: nx3 numpy array of all joint angles of joint 4, 6, 7

        Returns:
            joints_123 = nx3 numpy array of all joint angles of joint 1, 2 ,3
        """
        # Get R^3_0, by composing the transformations in reverse.
        joints_123 = np.zeros((4,3)) 
        if not debug:
            return joints_123
        else:
            return joints_123, []
    
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

                    q_as_part = self.sort_joints(q_as[i:idx, :], col+1)
                    q_as[i:idx, :] = q_as_part
        else: 
            q_as = q[q[:, -1].argsort()]
        return q_as

def main(): 
    
    # fk solution code
    fk = FK()

    # input joints  
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = -np.pi/2
    q5 = 0
    q6 = np.pi/2
    q7 = np.pi/4
        
    q_in  = np.array([q1, q2, q3, q4, q5, q6, q7])
    [_, T_fk] = fk.forward(q_in)

    # input of IK class
    target = {'R': T_fk[0:3, 0:3], 't': T_fk[0:3, 3]}
    ik = IK()
    q, frames = ik.panda_ik(target, debug=True)
    
    # verify IK solutions 
    for i in range(q.shape[0]):
        [_, T_ik] = fk.forward(q[i, :])
        print('Matrix difference = ')
        print(T_fk - T_ik)
        print()

    print(q)

if __name__ == '__main__':
    main()













