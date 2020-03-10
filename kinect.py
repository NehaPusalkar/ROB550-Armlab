"""!
Class to represent the kinect.
"""

import cv2
import numpy as np
from PyQt4.QtGui import QImage
import freenect
import os
import math
script_path = os.path.dirname(os.path.realpath(__file__))

class Kinect():
    """!
    @brief      This class describes a kinect.
    """

    def __init__(self):
        """!
        @brief      Constructs a new instance.
        """
        self.VideoFrame = np.array([])
        self.DepthFrameRaw = np.array([]).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((480,640,3)).astype(np.uint8)
        self.DepthFrameRGB=np.array([])

        """initialize kinect & turn off auto gain and whitebalance"""
        freenect.sync_get_video_with_res(resolution=freenect.RESOLUTION_HIGH)
        # print(freenect.sync_set_autoexposure(False))
        freenect.sync_set_autoexposure(False)
        # print(freenect.sync_set_whitebalance(False))
        freenect.sync_set_whitebalance(False)
        """check depth returns a frame, and flag kinectConnected"""
        if(freenect.sync_get_depth_with_res(format = freenect.DEPTH_11BIT) == None):
            self.kinectConnected = False
        else:
            self.kinectConnected = True
        print(self.kinectConnected)
        # mouse clicks & calibration variables
        self.depth2rgb_affine = np.float32([[1,0,0],[0,1,0]])
        self.cam_matrix_inv = np.float32([[1,0,0],[0,1,0],[0,0,1]])
        self.ex_matrix = np.float32([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        self.kinectCalibrated = False
        self.last_click = np.array([0,0])
        self.new_click = False
        self.rgb_click_points = np.zeros((10,2),int)
        self.depth_click_points = np.zeros((10,2),int)

        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def toggleExposure(self, state):
        """!
        @brief      Toggle auto exposure

        @param      state  False turns off auto exposure True turns it on
        """
        if state == False:
            freenect.sync_get_video_with_res(resolution=freenect.RESOLUTION_HIGH)
            # print(freenect.sync_set_autoexposure(False))
            freenect.sync_set_autoexposure(False)
            # print(freenect.sync_set_whitebalance(False))
            freenect.sync_set_whitebalance(False)
        else:
            freenect.sync_get_video_with_res(resolution=freenect.RESOLUTION_HIGH)
            # print(freenect.sync_set_autoexposure(True))
            freenect.sync_set_autoexposure(True)
            # print(freenect.sync_set_whitebalance(True))
            freenect.sync_set_whitebalance(True)

    def captureVideoFrame(self):
        """!
        @brief Capture frame from Kinect, format is 24bit RGB
        """
        if(self.kinectConnected):
           self.VideoFrame = freenect.sync_get_video_with_res(resolution=freenect.RESOLUTION_HIGH)[0]
        else:
            self.loadVideoFrame()
        self.processVideoFrame()


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        self.block_and_color_detect()

    def captureDepthFrame(self):
        """!
        @brief Capture depth frame from Kinect, format is 16bit Grey, 10bit resolution.
        """
        if(self.kinectConnected):
            if(self.kinectCalibrated):
                self.DepthFrameRaw = self.registerDepthFrame(freenect.sync_get_depth_with_res(format = freenect.DEPTH_11BIT)[0])
            else:
                self.DepthFrameRaw = freenect.sync_get_depth_with_res(format = freenect.DEPTH_11BIT)[0]
        else:
            self.loadDepthFrame()

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[...,0] = self.DepthFrameRaw
        self.DepthFrameHSV[...,1] = 0x9F
        self.DepthFrameHSV[...,2] = 0xFF
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread(script_path + "/data/rgb_image.png",cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread(script_path + "/data/raw_depth.png",0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (640, 480))
            img = QImage(frame,
                             frame.shape[1],
                             frame.shape[0],
                             QImage.Format_RGB888
                             )
            return img
        except:
            return None

    def convertQtDepthFrame(self):
       """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
       try:
           img = QImage(self.DepthFrameRGB,
                            self.DepthFrameRGB.shape[1],
                            self.DepthFrameRGB.shape[0],
                            QImage.Format_RGB888
                            )
           return img
       except:
           return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

                    TODO: Rewrite this function to take in an arbitrary number of coordinates and find the transform without
                    using cv2 functions

        @param      coord1  The coordinate 1
        @param      coord2  The coordinate 2

        @return     Affine transform between coordinates.
        """
        A = []
        B = []
        for point in coord2:
            A.append([point[0], point[1], 1, 0, 0, 0])
            A.append([0, 0, 0, point[0], point[1], 1])
        for point in coord1:
            B.append(point[0])
            B.append(point[1])
        return np.reshape(np.dot(np.linalg.pinv(np.array(A)), np.array(B)),(2,3))
        #pts1 = coord1[0:3].astype(np.float32)
        #pts2 = coord2[0:3].astype(np.float32)
        #print(cv2.getAffineTransform(pts1, pts2))
        #return cv2.getAffineTransform(pts1, pts2)

    def get_xyz_in_world(self, rgb_click_point):
        depth = self.DepthFrameRaw[rgb_click_point[1], rgb_click_point[0]]
        depth = 0.1236 * np.tan(depth/2842.5 + 1.1863) * 1000
        xyz_in_cam = depth * np.dot(self.cam_matrix_inv, np.append(rgb_click_point, 1))
        xyz_in_world = xyz_in_cam.reshape((3,1)) - self.ex_matrix[:,3].reshape((3,1))
        xyz_in_world = np.dot(np.linalg.inv(self.ex_matrix[0:3, 0:3]), xyz_in_world)
        return list(xyz_in_world.reshape(3,))

    def registerDepthFrame(self, frame):
        """!
        @brief      Transform the depth frame to match the RGB frame

                    TODO: Using an Affine transformation, transform the depth frame to match the RGB frame using
                    cv2.warpAffine()

        @param      frame  The frame

        @return     { description_of_the_return_value }
        """
        return cv2.warpAffine(frame, self.depth2rgb_affine, (np.shape(frame)[1], np.shape(frame)[0]))

    def loadCameraCalibration(self):
        """!
        @brief      Load camera intrinsic matrix from file.

        @param      file  The file
        """
        cam_matrix = np.array([[ 518.78051904,    0. ,         318.6431452 ],
                                [   0.       ,   518.6592693  , 267.11464023],
                                [   0.        ,    0.          ,  1.        ]])
        coeff = np.array([2.49214268e-01, -8.25220241e-01, 1.64661e-03,  -1.79181e-03, 1.131698341e+00])
        affine_matrix = np.array([[  9.29346844e-1,  -3.23494980e-03,   11.2347976],
                                  [  1.48233033e-03,   8.74361534e-01,   31.8750435]])
        return cam_matrix, coeff, affine_matrix

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        # contours = cv2.findContours(self.DepthFrameRaw, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE)
        # perimeter = cv2.arcLength(contours,True)
        # epsilon = 0.1*cv2.arcLength(contours,True)
        # approx = cv2.approxPolyDP(contours,epsilon,True)

        pass
    
    def block_and_color_detect(self):
        colors = {'red': [0,8,200,255,75,115], 'green':[40,70,100,255,45,85], 'blue':[100,130,100,200,60,90],
         'orange': [8,18,200,255,90,150], 'purple':[140,170,100,170,45,75], 'pink': [170,180,180,250,125,165],
         'yellow' : [20,30,200,255,100,255], 'black': [5,30,100,210,20,60]}

        colors_center = {'red': [2,227.5,85],'red2':[178, 227.5, 85], 'green' : [55,177.5,55], 'blue' : [115,150,75],
                        'orange' : [10,250,105], 'purple' :[155,135,60], 'pink': [175,215,145],
                        'yellow' : [25,227.5,145], 'black': [17.5,155,40]}

        #frame =cv2.imread("./data/rgb_image.png",cv2.IMREAD_UNCHANGED)
        #depth_frame = cv2.imread("./data/raw_depth.png",0).astype(np.uint8)
        frame = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2BGR)
        frame = cv2.medianBlur(frame, 5)
        frame = cv2.GaussianBlur(frame, (5,5), 0)
        video_frame = cv2.resize(frame, (640, 480))
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)
        if(self.DepthFrameRaw.size != 0):
            depth_frame = self.DepthFrameRaw.astype(np.uint8)
        else:
            return
        depth_frame = cv2.medianBlur(depth_frame, 3)
        th1 = cv2.inRange(depth_frame, 158, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        th1 = cv2.erode(th1, kernel, iterations=2)
        th1 = cv2.dilate(th1, kernel, iterations=1)
        binary = cv2.Canny(th1, 10, 90)
        image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # canvas = np.zeros((480,640,3)).astype(np.uint8)
        # canvas[...,0] = depth_frame
        # canvas[...,1] = depth_frame
        # canvas[...,2] = depth_frame
        block_center = []
        affine_matrix = np.array([[ 9.29346844e-1,  -3.23494980e-03,   11.2347976],
                                 [1.48233033e-03,   8.74361534e-01,   31.8750435]])
        if(len(contours)!=0):
            for contour in contours:
                #cv2.drawContours(canvas, contours, -1, (255,0,255), 3)
                perimeter = cv2.arcLength(contour,True)
                epsilon = 0.1 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                overlap = False
                
                if(len(approx)==4):
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cur_center = np.sum(approx, axis=0)/len(approx)
                    for center in block_center:
                        if(np.sum((center - cur_center)**2)<50):
                            overlap = True
                            break
                    if(not overlap and cv2.contourArea(approx) > 200):
                        block_center.append(cur_center)
                        if self.kinectCalibrated:
                            box_in_rgb = box
                        else:
                            box_h = np.concatenate((box.T, np.ones([1, 4])), axis=0)
                            box_in_rgb = np.dot(affine_matrix, box_h).T
                        box_in_rgb = ([2, 2.1333]*box_in_rgb[:][0:4]).astype(int)
                        cv2.drawContours(self.VideoFrame, [box_in_rgb], -1, (255,0,255), 2)

        num = len(block_center)
        block_center = np.array(block_center).reshape(num,2).T
        block_center_h = np.concatenate((block_center, np.ones([1, num])), axis=0)
        block_center_in_rgb = np.dot(affine_matrix, block_center_h).T
        font = cv2.FONT_HERSHEY_SIMPLEX
        for center in block_center_in_rgb:
            min_loss = 200000
            x=int(center[0])
            y=int(center[1])
            h = int(video_frame[y][x][0])
            s = int(video_frame[y][x][1])
            v = int(video_frame[y][x][2])
            #print("=======")
            for key in colors_center:
                h_diff = abs(h - colors_center[key][0])
                s_diff = abs(s - colors_center[key][1])
                v_diff = abs(v - colors_center[key][2])
                
                loss = math.sqrt(0.8 * (h_diff**2) + 0.05 * (s_diff**2) + 0.15 * (v_diff**2)) 
                #loss = math.sqrt(1 * (h_diff**2) + 1 * (s_diff**2) + 1 * (v_diff**2)) 
                #print(key+':'+str(loss))
                if (loss < min_loss):
                    min_loss = loss
                    color_detected = key
            output_h = "{}".format(color_detected)
            cv2.putText(self.VideoFrame, output_h, (2*x, int(2.1333*y) - 20), font, 1, (0, 0, 0))
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.imshow("image",canvas)
