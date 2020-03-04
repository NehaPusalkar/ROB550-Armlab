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

link = np.array([0.03953, 0.0975, 0.09879, 0]) # l1, l2, l3, l4
offset = np.array([0, 0.03468, 0, 0]) #n1, n2, n3, n4

dh_params = [[0, np.pi/2, link[0], np.pi/2],
             [link[1], 0, 0, np.pi/2],
             [link[2] + offset[1], 0, 0, -np.pi/2],
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
    
    item = dh_params[link]
    return get_transform_from_dh(item[0], item[1] + joint_angles[link], item[2], item[3])


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
    Ai = np.dot(Rotxalpha_Matrix(alpha), Transxa_Matrix(a))
    Ai = np.dot(Ai, Transzd_Matrix(d)) 
    Ai = np.dot(Ai, Rotztheta_Matrix(theta))
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

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """

    phi = get_euler_angles_from_T(T)[1]
    x = T[0, -1]
    y = T[1, -1]
    z = T[2, -1]
    return np.array([x, y, z, phi])


def FK_pox(joint_angles):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    return np.array([0, 0, 0, 0])

def to_s_matrix(w,v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    return np.zeros((4,4))
