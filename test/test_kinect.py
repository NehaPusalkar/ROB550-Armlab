#!/usr/bin/python3
"""!
Test the kinect

TODO: Use this file and modify as you see fit to test kinect
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(script_path + '/../')
import sys
import cv2
import numpy as np
from kinect import *

font = cv2.FONT_HERSHEY_SIMPLEX

def mouse_callback(event, x, y, flags, param):
    img = video_frame
    _video = cv2.medianBlur(kinect.VideoFrame, 3)
    hsv = cv2.cvtColor(_video, cv2.COLOR_RGB2HSV)
    r = img[y][x][2]
    g = img[y][x][1]
    b = img[y][x][0]
    h = hsv[y][x][0]
    s = hsv[y][x][1]
    v = hsv[y][x][2]
    output_rgb = "R:{}, G:{}, B:{} ".format(r, g, b)
    output_hsv = "H:{}, S:{}, V:{}".format(h, s, v)
    tmp = video_frame.copy()
    cv2.putText(tmp,output_rgb, (10, 20), font, 1, (255, 0, 0), 2)
    cv2.putText(tmp,output_hsv, (10, 40), font, 1, (255, 0, 0), 2)
    cv2.imshow('Video', tmp)

def mouse_callback_depth(event, x, y, flags, param):
    img = depth_frame
    b = img[y][x]
    output_rgb = "V:{}".format(b)
    tmp = depth_frame.copy()
    cv2.putText(tmp,output_rgb, (10, 20), font, 0.5, (0, 0, 0))
    cv2.imshow('Depth', tmp)

kinect = Kinect()

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')


# cap = cv2.VideoFrame(1)
while True:
    kinect.captureVideoFrame()
    kinect.captureDepthFrame()
    kinect.ColorizeDepthFrame()
    # depth_frame = cv2.cvtColor(kinect.DepthFrameRGB, cv2.COLOR_RGB2GRAY)
    video_frame = kinect.VideoFrame
    video_frame = cv2.cvtColor(kinect.VideoFrame, cv2.COLOR_RGB2BGR)
    # depth_frame = cv2.medianBlur(depth_frame, 5)
    # ret, th1 = cv2.threshold(depth_frame, 210, 255, cv2.THRESH_BINARY)
    # binary = cv2.Canny(th1, 10, 90)
    # image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # canvas = np.zeros((480,640,3)).astype(np.uint8)
    # canvas[...,0] = depth_frame
    # canvas[...,1] = depth_frame
    # canvas[...,2] = depth_frame
    # if(len(contours)!=0):
    #     num = 0
    #     for contour in contours:
    #         print(num)
    #         cv2.drawContours(binary, contours, -1, 255*num/len(contours), 3)
    #         perimeter = cv2.arcLength(contour,True)
    #         epsilon = 0.1 * perimeter
    #         approx = cv2.approxPolyDP(contour, epsilon, True)
    #         cv2.drawContours(canvas, approx, -1, (255,0,255), 3)
    #         num = num + 1

    #video_frame =  cv2.cvtColor(kinect.VideoFrame, cv2.COLOR_RGB2HSV)
    #hsv_frame = cv2.cvtColor(kinect.VideoFrame, cv2.COLOR_RGB2HSV)
    #output_frame = cv2.inRange(hsv_frame,(110,100,100), (120,255,255))
    #kernel = np.ones((4,4),np.uint8)
    #output_frame = cv2.erode(output_frame, kernel)
    #output_frame = cv2.dilate(output_frame, kernel, iterations=2)
    kinect.block_and_color_detect()
    #cv2.imshow('Depth', binary)
    #cv2.imshow('Video', video_frame)
    #kinect.blockDetector()
    #cv2.setMouseCallback("Video", mouse_callback)
    #cv2.imshow('Output', output_frame)
    k = cv2.waitKey(10)
    if(k == 27): # 'Esc' key
        break
    elif(k == 97): # 'a' key
        kinect.toggleExposure(True)
    elif(k==120): # 'x' key
        kinect.toggleExposure(False)
    elif(k==112): # 'p' key
        cv2.imwrite(script_path + "/data/rgb_image.png", cv2.cvtColor(kinect.VideoFrame, cv2.COLOR_RGB2BGR))
        depth = kinect.DepthFrameRaw.astype(np.uint16) * 256
        cv2.imwrite(script_path + "/data/raw_depth.png", depth)
        print(script_path + "/data/raw_depth.png")
        print("picture taken\n")
    elif(k != -1):
        print("Options: \n'Esc' to Quit, \n'a' to turn on autogain \n'x' to turn off autogain \n'p' to take an image")
