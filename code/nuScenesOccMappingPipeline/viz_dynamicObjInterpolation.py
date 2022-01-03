import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils.interUtils as interUtils
# from pyquaternion import Quaternion

plt.close("all")
#============================================================================#
# 2D Pose interpolation
#============================================================================#
def plot2DPose(p,color=None):
    R = interUtils.rotMat(p[2])
    if (color==None):
        plt.plot([p[0],p[0]+R[0,0]],
                 [p[1],p[1]+R[1,0]],"r")
        plt.plot([p[0],p[0]+R[0,1]],
                 [p[1],p[1]+R[1,1]],"g")
    else:
        plt.plot([p[0],p[0]+R[0,0]],
                 [p[1],p[1]+R[1,0]],color)
        plt.plot([p[0],p[0]+R[0,1]],
                 [p[1],p[1]+R[1,1]],color)

# init poses
p0 = np.array([0.5,-0.5,80/180*np.pi])
p1 = np.array([-0.5,4,140/180*np.pi])

# compute relative pose in homogeneous coordinates
T1 = interUtils.pose2HomogTrafo_2D(p1)    
T0 = interUtils.pose2HomogTrafo_2D(p0)    
T10 = np.dot( np.linalg.inv(T0), T1)

# represent all poses in p0 coordinate frame
p1 = interUtils.homogTrafo2Pose_2D(T10)
p0 = np.array([0,0,0])

t0 = 0.
t1 = 1.

# interpolate poses
p01, t01 = interUtils.interpolate2DPoses(p0, p1, t0, t1, numInterPts=10, minDist=0.1)

# transform them into reletive coordinates


# viz poses
plt.figure()
lim = 5
plt.plot([0,0],[-lim,lim],"gray")
plt.plot([-lim,lim],[0,0],"gray")
for i in range(p01.shape[0]):
    plot2DPose(p01[i,:],"gray")
plot2DPose(p0)
plot2DPose(p1)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])

#============================================================================#
# 3D Pose interpolation
#============================================================================#
# def plot3DPose(T):
#     plt.plot([T[0,3],T[0,3]+T[0,0]],
#              [T[1,3],T[1,3]+T[1,0]],
#              [T[2,3],T[2,3]+T[2,0]],"r")
#     plt.plot([T[0,3],T[0,3]+T[0,1]],
#              [T[1,3],T[1,3]+T[1,1]],
#              [T[2,3],T[2,3]+T[2,1]],"g")
#     plt.plot([T[0,3],T[0,3]+T[0,2]],
#              [T[1,3],T[1,3]+T[1,2]],
#              [T[2,3],T[2,3]+T[2,2]],"b")

# # init poses
# T0 = Quaternion(axis=[0, -1, 1], angle=0.).transformation_matrix
# T0[:3,3] = np.array([0,0,0])
# T1 = Quaternion(axis=[1, 0, 0], angle=0.).transformation_matrix
# T1[:3,3] = np.array([2,2,0])

# # interpolate poses
# t0 = 0.
# t1 = 1.
# T01, t01 = interUtils.interpolate3DPoses(T0, T1, t0, t1, numInterPts=10, minDist=0.3)

# # viz poses
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for T in T01:
#     plot3DPose(T)

# ax.view_init(elev=23., azim=-116.)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# lim = 2
# ax.set_xlim([-lim,lim])
# ax.set_ylim([-lim,lim])
# ax.set_zlim([-lim,lim])
