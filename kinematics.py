"""!
Implements Forward and backwards kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
import math

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

link = np.array([0.04853, 0.096, 0.09879, 0.125, 0.137]) # l1, l2, l3, l4 101.44 98.44
offset = np.array([0.07, 0.03234, 0, 0]) #base, n2, n3, n4
a = math.atan2(link[1], offset[1])
dh_params = [[0, np.pi/2, link[0]+offset[0], np.pi/2],
             [math.sqrt(link[1]**2 + offset[1]**2), 0, 0, a],
             [link[2], 0, 0, -a],
             [0, np.pi/2, 0, np.pi/2]]

def Rotztheta_Matrix(theta):
    M = np.eye(4)
    M[0][0] = math.cos(theta)
    M[1][1] = math.cos(theta)
    M[0][1] = -math.sin(theta)
    M[1][0] = math.sin(theta)
    return M

def Rotxalpha_Matrix(alpha):
    M = np.eye(4)
    M[2][2] = math.cos(alpha)
    M[1][1] = math.cos(alpha)
    M[1][2] = -math.sin(alpha)
    M[2][1] = math.sin(alpha)
    return M

def Transzd_Matrix(d):
    M = np.eye(4)
    M[2][-1] = d
    return M

def Transxa_Matrix(a):
    M = np.eye(4)
    M[0][-1] = a
    return M

def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    A = []
    i = 0

    for item in dh_params:
        if i <= link:
            A.append(get_transform_from_dh(item[0], item[1], item[2], item[3] + joint_angles[i]))
        else:
            break
        i += 1

    H = np.eye(4)
    for item in A:
        H = np.dot(H, item)
    return H


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    # Ai = np.dot(Rotxalpha_Matrix(alpha), Transxa_Matrix(a))
    # Ai = np.dot(Ai, Transzd_Matrix(d))
    # Ai = np.dot(Ai, Rotztheta_Matrix(theta))
    Ai = np.dot(Rotztheta_Matrix(theta), Transzd_Matrix(d))
    Ai = np.dot(Ai, Transxa_Matrix(a))
    Ai = np.dot(Ai, Rotxalpha_Matrix(alpha))
    return Ai

def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    sy = math.sqrt(T[0,0] * T[0,0] +  T[1,0] * T[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(T[2,1] , T[2,2])
        y = math.atan2(-T[2,0], sy)
        z = math.atan2(T[1,0], T[0,0])
    else :
        x = math.atan2(-T[1,2], T[1,1])
        y = math.atan2(-T[2,0], sy)
        z = 0

    return np.array([x, y, z], dtype = np.float32)

def get_phi_from_angles(joint_angles):
    return joint_angles[1] + joint_angles[2] + joint_angles[3]

def get_pose_from_T(T, joint_angles):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """

    phi = get_phi_from_angles(joint_angles)
    x = T[0, -1]
    y = T[1, -1]
    z = T[2, -1]
    return np.array([x, y, z, phi])

def map_angle(angle):
    if(angle > math.pi):
        return angle - 2*math.pi*int(((angle+math.pi)/math.pi)/2)
    elif(angle < -math.pi):
        return angle - 2*math.pi*int(((angle-math.pi)/math.pi)/2)
    else:
        return angle

def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    x = pose[0]
    y = pose[1]
    z = pose[2]
    phi = pose[3]
    base = dh_params[0][2]
    d = math.sqrt(x**2 + y**2)
    alpha = math.pi/2 - dh_params[1][3]
    l = math.sqrt((base - z)**2 + d**2)
    l1 = dh_params[1][0]
    l2 = dh_params[2][0]
    gamma = math.atan2(d,(base-z))

    psi = math.acos((l**2 + l1**2 - l2**2)/(2*l*l1))
    beta = math.acos((l1**2 + l2**2 - l**2)/(2*l1*l2))

    #solution 1&2
    theta_0_0 = - math.atan2(x, y)
    theta_2_0 = -(math.pi/2 + alpha - beta)
    theta_1_0 = -(math.pi - alpha - psi - gamma)
    theta_3_0 = phi - theta_1_0 - theta_2_0

    theta_2_1 = (math.pi*2 - beta) - math.pi/2 - alpha
    theta_1_1 = -math.pi + alpha + gamma - psi
    theta_3_1 = phi - theta_1_1 - theta_2_1
    #solution 3&4
    theta_0_1 = map_angle(math.pi - math.atan2(x, y))

    theta_2_2 = math.pi*2 - beta - alpha - math.pi/2
    theta_1_2 =  math.pi + alpha - gamma - psi
    theta_3_2 = phi - theta_1_2 - theta_2_2

    theta_2_3 = -(math.pi + alpha - math.pi/2 - beta)
    theta_1_3 = math.pi + alpha + psi - gamma
    theta_3_3 = phi - theta_1_3 - theta_2_3

    return np.array([[theta_0_0, theta_1_0, theta_2_0, theta_3_0], \
                     [theta_0_0, theta_1_1, theta_2_1, theta_3_1], \
                     [theta_0_1, theta_1_2, theta_2_2, theta_3_2], \
                     [theta_0_1, theta_1_3, theta_2_3, theta_3_3]])
