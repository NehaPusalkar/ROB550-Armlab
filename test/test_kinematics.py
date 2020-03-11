#!/usr/bin/python3
"""!
Test kinematics

TODO: Use this file and modify as you see fit to test kinematics.py
"""

import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../'))
from kinematics import *
from copy import deepcopy
import math

passed = True
vclamp = np.vectorize(clamp)

link = np.array([0.04853, 0.096, 0.09879, 0.1]) # l1, l2, l3, l4 101.44 98.44
offset = np.array([0.07, 0.03234, 0, 0]) #base, n2, n3, n4
a = math.atan2(link[1], offset[1])
dh_params = [[0, np.pi/2, link[0]+offset[0], np.pi/2],
             [math.sqrt(link[1]**2 + offset[1]**2), 0, 0, a],
             [link[2], 0, 0, -a],
             [0, np.pi/2, 0, np.pi/2]]

fk_angles = [
    [0.0,           0.0,            0.0,            0.0],
    [np.pi * 0.1,   0.0,            np.pi / 2,      0.0],
    [np.pi * 0.25,  np.pi / 2,      -np.pi / 2,     np.pi / 2],
    [np.pi * 0.4,   np.pi / 2,      -np.pi / 2,     0.0],
    [np.pi * 0.55,  0,              0,              0],
    [np.pi * 0.7,   0.0,            np.pi / 2,      0.0],
    [np.pi * 0.85,  np.pi / 2,      -np.pi / 2,     np.pi / 2],
    [np.pi/2,         0,      np.pi / 2,     0.0],
    [0.0,           np.pi / 2,      np.pi / 2,      0.0],
    [np.pi / 2,     -np.pi / 2,     np.pi / 2,      0.0]]

print('Test FK')
fk_poses = []
for joint_angles in fk_angles:
    print('Joint angles:', joint_angles)
    for i, _ in enumerate(joint_angles):
        # print(FK_dh(deepcopy(dh_params), joint_angles, i))
        pose = get_pose_from_T(FK_dh(deepcopy(dh_params), joint_angles, i), joint_angles)
        print('Link {} pose: {}'.format(i, pose))
        if i == len(joint_angles) - 1:
            fk_poses.append(pose)
    print()

print('Test IK')
for pose, angles in zip(fk_poses, fk_angles):
    matching_angles = False
    print('Pose: {}'.format(pose))
    options = IK_geometric(deepcopy(dh_params), pose)
    for i, joint_angles in enumerate(options):
        print('Option {}: {}'.format(i, joint_angles))
        compare = vclamp(joint_angles - angles)
        if np.allclose(compare, np.zeros_like(compare), rtol=1e-3, atol=1e-4):
            print('Option {} matches angles used in FK'.format(i))
            matching_angles = True
    if not matching_angles:
        print('No match to the FK angles found!')
        passed = False
    print()
