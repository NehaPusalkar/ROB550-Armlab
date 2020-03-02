"""!
Class to represent the kinect.
"""

import cv2
import numpy as np
from PyQt4.QtGui import QImage
import freenect
import os
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
        cv2.drawContours(self.VideoFrame,self.block_contours,-1,(255,0,255),3)


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
        print(xyz_in_cam.shape)
        print(xyz_in_world.shape)
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
        image = self.VideoFrame
        blur = cv2.GaussianBlur(image,(5,5),5)
        cv2.namedWindow("first_image",cv2.WINDOW_NORMAL)
        cv2.imshow("first image",image)
# convert to HSV

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv",hsv)
        # set limits on values for that color
        lower_val =np.array([70, 120, 0])
        upper_val = np.array([255,255,5])

        lower_red = np.array([70,120,170])
        upper_red = np.array([255,255,180])

        mask1 = cv2.inRange(hsv,lower_val,upper_val)
        mask2 = cv2.inRange(hsv,lower_red,upper_red)

        mask = mask1 + mask2

        
        output_img = image.copy()
        output_img[np.where(mask==0)] = 0

        output_hsv = hsv.copy()
        output_hsv[np.where(mask==0)] = 0
        cv2.imshow("image",cv2.WINDOW_NORMAL)
        cv2.imshow("image",output_img)
        cv2.namedWindow("hsv_image", cv2.WINDOW_NORMAL)
        cv2.imshow("hsv_image",output_hsv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

       

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
