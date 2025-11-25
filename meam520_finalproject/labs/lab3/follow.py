import sys
import numpy as np
import rospy
from math import cos, sin, pi
import matplotlib.pyplot as plt
import geometry_msgs

from core.interfaces import ArmController
from core.utils import time_in_seconds

from lib.IK_velocity import IK_velocity
from lib.calculateFK import FK

class JacobianDemo():
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function
    """
    active = False # When to stop commanding arm
    start_time = 0 # start time
    dt = 0.03 # constant for how to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher('/vis/trace', geometry_msgs.msg.PointStamped, queue_size=10)
    counter = 0
    x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
    last_iteration_time = None
    last_pose = x0
    errors = list()
    speeds = list()
    last_xdes = np.array([0.307, 0, 0.487])
        

    ##################
    ## TRAJECTORIES ##
    ##################

    def eight(t,fx=1,fy=2,rx=.15,ry=.1):
        """
        Calculate the position and velocity of the figure 8 trajector

        Inputs:
        t - time in sec since start
        fx - frequecny in rad/s of the x portion
        fy - frequency in rad/s of the y portion
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """

        # Lissajous Curve
        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
        xdes = x0 + np.array([rx*sin(fx*t),ry*sin(fy*t),0])
        vdes = np.array([rx*fx*cos(fx*t), ry*fy*cos(fy*t),0])
        JacobianDemo.last_pose = xdes
        return xdes, vdes

    def ellipse(t,f=1,ry=.15,rz=.1):
        """
        Calculate the position and velocity of the figure ellipse trajector

        Inputs:
        t - time in sec since start
        f - frequency in rad/s of the trajectory
        ry - radius in m of the y portion
        rz - radius in m of the z portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """

        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position

        ## STUDENT CODE GOES HERE!
        # rads/s * t = progress
        rads = (t * f) + 3*pi/2  # Add in offset to start at the bottom of the ellipse
        ## No slope or halfway point, as the start and end point are the same.
        
        xdes = x0 + np.array([0, ry*cos(rads), (rz*sin(rads))+rz])
        # We could make the above simpler by using trig identites, but it wouldn't make a difference here
        # We could use t*f instead of rads, and switch the sin and cos due to the offsets

        # To get this, we use the simplified expression.
        # The derivative was taken w.r.t t
        vdes = np.array([0, f*ry*cos(f*t), f*rz*sin(f*t)])

        ## END STUDENT CODE
        JacobianDemo.last_pose = xdes
        return xdes, vdes

    def line(t,f=1.25,L=.2):
        """
        Calculate the position and velocity of the line trajector

        Inputs:
        t - time in sec since start
        f - frequency in Hz of the line trajectory
        L - length of the line in meters
        
        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """
        ## STUDENT CODE GOES HERE
        x0 = JacobianDemo.x0
        endpoint = x0 + np.array([0, 0, L])
        p = 1/f
        position = 2*np.abs((t/p) - np.floor((t/p) + 0.5))
        
        halfway_point = (1/f)/2  # Period/2
        # When reps is 0.5, we are at the endpoint, we need to go back
        xdes = (1- position)*x0 + (position)*endpoint
        # Our velocity is the time derivative of the positions
        if (t%(1/f)) > halfway_point:  # On the way back
            vdes = (x0 - endpoint)*f
        else:
            vdes = (endpoint - x0)*f
        # Below is the sine wave method. This results in peak velocities that are too high.
        #position = 1 - cos(t*f*2*pi)
        #xdes = (1- position)*x0 + (position)*endpoint
        #vdes = (-2*pi*f*x0*sin(2*pi*f*t) + 2*pi*f*endpoint*sin(2*pi*f*t))
        JacobianDemo.last_pose = xdes
        return xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.point.x = JacobianDemo.last_pose[0]
        msg.point.y = JacobianDemo.last_pose[1]
        msg.point.z = JacobianDemo.last_pose[2]
        self.point_pub.publish(msg)

    ################
    ## CONTROLLER ##
    ################

    def follow_trajectory(self, state, trajectory):

        if self.active:

            try:
                t = time_in_seconds() - self.start_time

                # get desired trajectory position and velocity
                xdes, vdes = trajectory(t)

                # get current end effector position
                q = state['position']
                joints, T0e = self.fk.forward(q)
                x = (T0e[0:3,3])

                # Compute error in position:
                self.errors.append(np.linalg.norm(x-self.last_xdes))
                self.speeds.append(np.linalg.norm(vdes))
                self.last_xdes = xdes
                print("{:4f} & {:.4f}".format(np.mean(self.speeds[-1000:]), np.mean(self.errors[-1000:])))

                # First Order Integrator, Proportional Control with Feed Forward
                kp = 20
                v = vdes + kp * (xdes - x)

                # Velocity Inverse Kinematics
                dq = IK_velocity(q,v,np.array([np.nan,np.nan,np.nan]))

                # Get the correct timing to update with the robot
                if self.last_iteration_time == None:
                    self.last_iteration_time = time_in_seconds()
                
                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()
                
                new_q = q + self.dt * dq
                
                arm.safe_set_joint_positions_velocities(new_q, dq)
                
                # Downsample visualization to reduce rendering overhead
                self.counter = self.counter + 1
                if self.counter == 10:
                    self.show_ee_position()
                    self.counter = 0
                
                if len(self.errors) == 1001:
                    self.active = False

            except rospy.exceptions.ROSException:
                pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython jacobianDemo.py line\n\tpython jacobianDemo.py ellipse\n\tpython jacobianDemo.py eight")
        exit()

    rospy.init_node("follower")

    JD = JacobianDemo()

    if sys.argv[1] == 'line':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.line)
    elif sys.argv[1] == 'ellipse':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.ellipse)
    elif sys.argv[1] == 'eight':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.eight)
    else:
        print("invalid option")
        exit()

    arm = ArmController(on_state_callback=callback)

    # reset arm
    print("resetting arm...")
    arm.safe_move_to_position(arm.neutral_position())

    # start tracking trajectory
    JD.active = True
    JD.start_time = time_in_seconds()

    input("Press Enter to stop")
    JD.active = False
