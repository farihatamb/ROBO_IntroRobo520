from lib.calculateFK import FK
from core.interfaces import ArmController

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'start': -2.8973, 'stop': 2.8973},
    {'start': -1.7628, 'stop': 1.7628},
    {'start': -2.8973, 'stop': 2.8973},
    {'start': -3.0718, 'stop': -0.0698},
    {'start': -2.8973, 'stop': 2.8973},
    {'start': -0.0175, 'stop': 3.7525},
    {'start': -2.8973, 'stop': 2.8973}
 ]

# To visualize the joint limits, we will try to generate a volumetric
# representation by sampling many points within the configuation space.
points = list()
STEP_SIZE=0.75
ranges = [np.arange(limit['start'], limit['stop'], STEP_SIZE) for limit in limits]
print("Total Points: {}".format(np.prod([r.shape[0] for r in ranges])))

for q0 in np.arange(limits[0]['start'], limits[0]['stop'], STEP_SIZE):
    for q1 in np.arange(limits[1]['start'], limits[1]['stop'], STEP_SIZE):
        for q2 in np.arange(limits[2]['start'], limits[2]['stop'], STEP_SIZE):
            for q3 in np.arange(limits[3]['start'], limits[3]['stop'], STEP_SIZE):
                for q4 in np.arange(limits[4]['start'], limits[4]['stop'], STEP_SIZE):
                    for q5 in np.arange(limits[5]['start'], limits[5]['stop'], STEP_SIZE):
                        for q6 in np.arange(limits[6]['start'], limits[6]['stop'], STEP_SIZE):
                            jointPositions, _ = fk.forward(np.array([q0, q1, q2, q3, q4, q5, q6]))
                            points.append(jointPositions[7])
points = np.array(points)
hull = ConvexHull(points)


# We've included some very basic plotting commands below, but you can find
# more functionality at https://matplotlib.org/stable/index.html

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# TODO: update this with real results
# Convex Hull plotting code inspired by Stack Overflow https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
for s in hull.simplices:
    s = np.append(s, s[0])
    ax.plot(points[s, 0], points[s, 1], points[s, 2], "b-")
#ax.scatter(points[:, 0], points[:, 1], points[:, 2]) # plot the point (1,1,1)
ax.set_title("Convex Hull of reachable points")
plt.show()
