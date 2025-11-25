import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK, matrix_from_dh
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap

"""
from calcJacobian import calcJacobian
from calculateFK import FK, matrix_from_dh
from detectCollision import detectCollision
from loadmap import loadmap
"""


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = (
        lower + (upper - lower) / 2
    )  # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the
        target joint position and the current joint position

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint
        from the current position to the target position
        """

        ## STUDENT CODE STARTS HERE
        att_constant = 10.0

        diff = current - target
        dist = np.linalg.norm(diff)
        dist_threshold = 1.0
        if dist < dist_threshold:
            att_f = -att_constant * diff
        else:
            att_f = -dist_threshold * att_constant * diff / dist  # From textbook

        ## END STUDENT CODE
        assert not np.any(np.isnan(att_f))
        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3, 1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the
        obstacle and the current joint position

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position
        to the closest point on the obstacle box

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        p0 = 1.0
        rep_field_strength = 10.0
        shortest_dist, direction_vector = PotentialFieldPlanner.dist_point2box(
            current, obstacle
        )
        if shortest_dist > p0:
            rep_f = np.zeros((1, 3))
        else:
            rep_f = (
                rep_field_strength
                * ((1 / shortest_dist) - (1 / p0))
                * (1 / (shortest_dist**2))
                * -direction_vector
            )

        ## END STUDENT CODE
        #assert not np.any(np.isnan(rep_f))
        return rep_f

    def calc_all_pos_jacobians(q_in, joint_num):
        """
        Calculate the full Jacobian of the end effector in a given configuration
        :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
        :param joint_num: one indexed joint number
        :return: J - 3 x 7 matrix representing the Jacobian, where the first three
        rows correspond to the linear velocity and the last three rows correspond to
        the angular velocity, expressed in world frame coordinates
        """
        # From Lab 2
        d1 = 0.333
        d3 = 0.316
        a3 = 0.0825
        a4 = a3
        d5 = 0.384
        a6 = 0.088
        d7 = 0.210
        # DH parameters

        dhparams = np.array(
            [
                [0, -pi / 2, d1, q_in[0]],
                [0, pi / 2, 0, q_in[1]],
                [a3, pi / 2, d3, q_in[2]],
                [-a4, -pi / 2, 0, q_in[3]],
                [0, pi / 2, d5, q_in[4]],
                [a6, pi / 2, 0, q_in[5]],
                [0, 0, d7, q_in[6] - pi / 4],
            ]
        )
        J = np.zeros((3, joint_num + 1))

        A = [0 for _ in range(7)]
        for i in range(7):
            A[i] = matrix_from_dh(
                dhparams[i, 0], dhparams[i, 1], dhparams[i, 2], dhparams[i, 3]
            )
        T01 = A[0]
        T02 = A[0] @ A[1]
        T03 = A[0] @ A[1] @ A[2]
        T04 = A[0] @ A[1] @ A[2] @ A[3]
        T05 = A[0] @ A[1] @ A[2] @ A[3] @ A[4]
        T06 = A[0] @ A[1] @ A[2] @ A[3] @ A[4] @ A[5]
        T07 = A[0] @ A[1] @ A[2] @ A[3] @ A[4] @ A[5] @ A[6]
        TT = [T01, T02, T03, T04, T05, T06, T07]
        r = np.array([0, 0, 1])
        d0x = np.array([0, 0, 0.141])
        end_pos = TT[joint_num][:, 3]
        S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        d = np.array(
            [[end_pos[0] - d0x[0]], [end_pos[1] - d0x[1]], [end_pos[2] - d0x[2]]]
        )
        Jvx = S @ d
        J[0][0] = Jvx[0]
        J[1][0] = Jvx[1]
        J[2][0] = Jvx[2]

        for x in range(1, joint_num + 1):
            r = TT[x - 1][:, 2]
            d0x = TT[x - 1][:, 3]
            end_pos = TT[joint_num][:, 3]
            S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            d = np.array(
                [[end_pos[0] - d0x[0]], [end_pos[1] - d0x[1]], [end_pos[2] - d0x[2]]]
            )
            Jvx = S @ d
            J[0][x] = Jvx[0]
            J[1][x] = Jvx[1]
            J[2][x] = Jvx[2]

        assert not np.any(np.isnan(J))
        return J

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point

        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin * 0.5 + boxMax * 0.5
        p = np.array(p).reshape(-1, 3)

        # Get distance info from point to box boundary
        dx = np.amax(
            np.vstack(
                [boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]
            ).T,
            1,
        )
        dy = np.amax(
            np.vstack(
                [boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]
            ).T,
            1,
        )
        dz = np.amax(
            np.vstack(
                [boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]
            ).T,
            1,
        )

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter - p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum
        of forces (attactive, repulsive) on each joint.

        INPUTS:
        target - 3x7 numpy array representing the desired joint/end effector positions
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x7 numpy array representing the current joint/end effector positions
        in the world frame

        OUTPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros((3, 7))
        for j_idx in range(7):
            current_joint_pt = current[:, j_idx]
            current_target = target[:, j_idx]
            num_obstacles = len(obstacle)
            j_force = np.zeros((3,))
            # Add attractive force
            j_force += PotentialFieldPlanner.attractive_force(
                current_target, current_joint_pt
            )
            for obst_idx in range(num_obstacles):
                current_obstacle = obstacle[obst_idx]
                j_force += PotentialFieldPlanner.repulsive_force(current_obstacle, current_joint_pt).reshape(3,)
            joint_forces[:, j_idx] = j_force

        ## END STUDENT CODE

        return joint_forces

    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum
        of torques on each joint.

        INPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on each joint
        """

        ## STUDENT CODE STARTS HERE
        joint_torques = np.zeros(7)

        def pad_to_7(vector):
            v = np.zeros(7)
            for idx in range(vector.shape[0]):
                v[idx] = vector[idx]
            return v

        for joint_idx in range(7):
            jacobian = PotentialFieldPlanner.calc_all_pos_jacobians(
                q, joint_idx
            )  # 3xjoint_idx matrix
            # joint_idx x 1 = (3 x joint_idx)^T 3x1
            jt = jacobian.T @ joint_forces[:, joint_idx]
            joint_torques += pad_to_7(jt)
        ## END STUDENT CODE
        #assert not np.any(np.isnan(joint_torques))
        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance

    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal
        configuration

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task
        """

        ## STUDENT CODE STARTS HERE
        alpha = 0.05
        obstacles = map_struct.obstacles
        target_joints, _ = PotentialFieldPlanner.fk.forward(target)  # (8,3)
        target_joints = target_joints[:-1].T
        current_joints, _ = PotentialFieldPlanner.fk.forward(q)  # (8,3)
        current_joints = current_joints[:-1].T

        joint_forces = PotentialFieldPlanner.compute_forces(
            target_joints, obstacles, current_joints
        )
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)

        dq = alpha * joint_torques / np.linalg.norm(joint_torques)

        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Field Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes
        start - 1x7 numpy array representing the starting joint angles for a configuration
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles.
        """

        q_path = np.array([]).reshape(0, 7)
        current_q = start
        q_path = np.append(q_path, current_q.reshape(1, 7), axis=0)

        for i in range(self.max_steps):

            ## STUDENT CODE STARTS HERE

            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code

            # Compute gradient
            dq = self.compute_gradient(current_q, goal, map_struct)
            current_q = current_q + dq
            # Clip the q to be within bounds
            current_q = np.clip(
                current_q, PotentialFieldPlanner.lower, PotentialFieldPlanner.upper
            )

            q_path = np.append(q_path, current_q.reshape(1, 7), axis=0)

            # Termination Conditions
            closeness_condition = self.q_distance(goal, current_q) <= self.tol
            nan_condition = np.any(np.isnan(current_q))
            collision_condition = False
            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            joint_positions, _ = PotentialFieldPlanner.fk.forward(current_q)
            for obstacle in map_struct.obstacles:
                collisions = detectCollision(joint_positions[:-1], joint_positions[1:], obstacle)
                collision_condition = collision_condition or np.any(collisions)
            
            if closeness_condition or nan_condition or collision_condition:
                print(f"Closeness {closeness_condition} | NaN {nan_condition} | Collision {collision_condition}")
                break  # exit the while loop if conditions are met!

            
            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # We know that we haven't gotten close enough to our solution,
            # If we are wobbling around the same points, let's add a small random vector
            if self.q_distance(q_path[-2], current_q) < 1e-2:
                dq = np.random.randn(7)
                current_q = current_q + dq
                # Clip the q to be within bounds
                current_q = np.clip(
                    current_q, PotentialFieldPlanner.lower, PotentialFieldPlanner.upper
                )
                # Replace our last predicted pose with wiggle
                q_path[-1] = current_q

            ## END STUDENT CODE

        return q_path


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    planner = PotentialFieldPlanner()

    # inputs
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print(
            "iteration:", i, " q =", q_path[i, :], " error={error}".format(error=error)
        )

    print("q path: ", q_path)
