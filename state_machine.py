"""!
The state machine that implements the logic.
"""

import time
import numpy as np
import csv
import cv2
from trajectory_planner import TrajectoryPlanner
from apriltag import apriltag

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rexarm, planner, kinect):
        """!
        @brief      Constructs a new instance.

        @param      rexarm   The rexarm
        @param      planner  The planner
        @param      kinect   The kinect
        """
        self.rexarm = rexarm
        self.tp = planner
        self.kinect = kinect
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [0.0,           0.0,            0.0,            0.0],
            [np.pi * 0.1,   0.0,            np.pi / 2,      0.0],
            [np.pi * 0.25,  np.pi / 2,      -np.pi / 2,     np.pi / 2],
            [np.pi * 0.4,   np.pi / 2,      -np.pi / 2,     0.0],
            [np.pi * 0.55,  0,              0,              0],
            [np.pi * 0.7,   0.0,            np.pi / 2,      0.0],
            [np.pi * 0.85,  np.pi / 2,      -np.pi / 2,     np.pi / 2],
            [np.pi,         np.pi / 2,      -np.pi / 2,     0.0],
            [0.0,           np.pi / 2,      np.pi / 2,      0.0],
            [np.pi / 2,     -np.pi / 2,     np.pi / 2,      0.0]]

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

                    This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rexarm":
            self.initialize_rexarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute_tp":
            self.execute_tp()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "record":
            self.record()

        if self.next_state == "record_one":
            self.record_one()

        if self.next_state == "play_back":
            self.play_back()

        if self.next_state == "check_cali":
            self.check_cali()

        if self.next_state == "stop_record":
            self.next_state = "initialize_rexarm"

        if self.next_state == "clear_record":
            self.clear_record()

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the Rexarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check Rexarm and restart program"
        self.current_state = "estop"
        self.rexarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.next_state = "idle"
        for wp in self.waypoints:
            # Ensure the correct number of joint angles
            full_wp = [0.0] * self.rexarm.num_joints
            full_wp[0:len(wp)] = wp
            if(self.next_state == "estop"):
                break
            self.rexarm.set_positions(full_wp)
            time.sleep(1)

    def execute_tp(self):
        """!
        @brief      Go through all waypoints with the trajectory planner.
        """
        self.status_message = "State: Execute TP - Executing Motion Plan with trajectory planner"
        self.current_state = "execute"
        self.next_state = "idle"
        # waypoints = []
        for wp in self.waypoints:
            full_wp = [0.0] * self.rexarm.num_joints
            full_wp[0:len(wp)] = wp
            # waypoints.append(full_wp)
            # TODO: Send the waypoints to the trajectory planner and break if estop
            tp = TrajectoryPlanner(self.rexarm)
            tp.set_initial_wp()
            tp.set_final_wp(full_wp)
            tp.go(5)
            if(self.next_state == "estop"):
                break

    def calibrate(self):
        """!
        @brief      Gets the calibration clicks
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        location_strings = ["lower left corner of board",
                            "upper left corner of board",
                            "upper right corner of board",
                            "lower right corner of board",
                            "center of shoulder motor"]

        detector = apriltag("tagStandard41h12")
        image = self.kinect.VideoFrame #1280X960
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections = detector.detect(image)
        while(len(detections) != 5):
            image = self.kinect.VideoFrame
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            print(detections)
            print("Dececting again...")
            detections = detector.detect(image)
            if(self.next_state == "estop"):
                return False
        print("Detection Done!")
        detected_points = []
        for detec in detections:
            if(detec['id'] in range(2,8)):
                detected_points.append(detec['center'])
        '''
        i = 0
        for j in range(10):
            #self.status_message = "Calibration - Click %s in RGB image" % location_strings[j]
            while (i <= j):
                if(self.kinect.new_click == True):
                    self.kinect.rgb_click_points[i] = self.kinect.last_click.copy()
                    i = i + 1
                    self.kinect.new_click = False
        self.status_message = "Calibration Depth"
        i = 0
        depth = []
        for j in range(10):
            #self.status_message = "Calibration - Click %s in depth image" % location_strings[j]
            while (i <= j):
                if(self.kinect.new_click == True):
                    self.kinect.depth_click_points[i] = self.kinect.last_click.copy()
                    depth.append(self.kinect.DepthFrameRaw[self.kinect.depth_click_points[i][1], self.kinect.depth_click_points[i][0]])
                    i = i + 1
                    self.kinect.new_click = False
        
        """TODO Perform camera calibration here"""
        print("camera calibration now\n")
        self.kinect.depth2rgb_affine = self.kinect.getAffineTransform(self.kinect.rgb_click_points, self.kinect.depth_click_points)
        self.kinect.kinectCalibrated = True

        print(self.kinect.depth2rgb_affine)
        depth = 0.1236 * np.tan(np.array(depth)/2842.5 + 1.1863)
        '''
        cam_matrix, coeff, affine_matrix = self.kinect.loadCameraCalibration()
        self.kinect.depth2rgb_affine = affine_matrix
        real = np.array([[-570/2, 0, 75], [-570/2, -570/2, 0],[-570/2, 570/2, 0],[570/2, 570/2, 0],[570/2, -570/2, 0]], dtype=np.float32)
        ##TODO check [x,y] or [y,x]
        print([0.5, 0.46875]*np.array(detected_points))
        ex_matrix = cv2.solvePnP(real, [0.5, 0.46875]*np.array(detected_points, dtype=np.float32), cam_matrix, coeff)
        R = cv2.Rodrigues(ex_matrix[1])[0]
        t = ex_matrix[2]
        cam_matrix_inv = np.linalg.inv(cam_matrix)
        cam_matrix_inv[2][2] = 1
        #self.kinect.ex_matrix = np.concatenate((np.linalg.inv(R), -t), axis=1)
        self.kinect.ex_matrix = np.concatenate((R, t), axis=1)
        self.kinect.cam_matrix_inv = cam_matrix_inv
        print(self.kinect.ex_matrix)
        print(self.kinect.cam_matrix_inv)
        self.kinect.kinectCalibrated = True
        self.status_message = "Calibration - Completed Calibration"
        time.sleep(1)

    def check_cali(self):
        self.current_state = "check_cali"
        self.next_state = "idle"
        while(not self.next_state=='estop'):
            if(self.kinect.new_click == True):
                rgb_click_point = self.kinect.last_click.copy()
                self.kinect.new_click = False
                depth = self.kinect.DepthFrameRaw[rgb_click_point[1], rgb_click_point[0]]
                depth = 0.1236 * np.tan(depth/2842.5 + 1.1863)
                xyz_in_cam = depth * np.dot(self.kinect.cam_matrix_inv, np.append(rgb_click_point, 1))
                xyz_in_cam_h = np.append(xyz_in_cam, 1)
                xyz_in_world = np.dot(self.kinect.ex_matrix, xyz_in_cam_h)
                print(xyz_in_world)

    def clear_record(self):
        self.status_message = "State: Clearing Record ..."
        self.current_state = "clear_record"
        self.next_state = "idle"
        open('recording.txt', 'w').close()

    def record_one(self):
        """!
        @brief     Task 2.2 Record Actions One Point
        """
        self.status_message = "State: Recording One...(Teaching...)"
        self.current_state = "record_one"
        self.next_state = "idle"
        with open('recording.txt', 'a') as record_file:
            csv_writer = csv.writer(record_file, delimiter = ',')
            csv_writer.writerow(self.rexarm.position_fb)

    def record(self):
        """!
        @brief     Task 2.2 Record Actions(Teach)
        """
        self.status_message = "State: Recording...(Teaching...)"
        self.current_state = "record"
        self.next_state = "idle"
        self.rexarm.disable_torque()
        with open('recording.txt', 'w') as record_file:
            while True:
                csv_writer = csv.writer(record_file, delimiter = ',')
                csv_writer.writerow(self.rexarm.position_fb)
                time.sleep(0.1)
                if self.next_state == "stop_record":
                    break
                if self.next_state == "estop":
                    break

    def play_back(self):
        """!
        @brief     Task 2.2 Play Back All Actions(Repeat)
        """
        self.status_message = "State: Playing...(Repeating...)"
        self.current_state = "play_back"
        self.next_state = "initialize_rexarm"
        self.rexarm.set_torque_limits([40 / 100.0] * self.rexarm.num_joints)
        self.rexarm.set_speeds_normalized_all(20 / 100.0)
        with open('recording.txt', 'r') as record_file:
            csv_reader = csv.reader(record_file, delimiter = ',')
            for row in csv_reader:
                pos = []
                for item in row:
                    pos.append(float(item))
                print(pos)
                self.rexarm.set_positions(pos)
                if self.next_state == "estop":
                    break
                time.sleep(1)


    def initialize_rexarm(self):
        """!
        @brief      Initializes the rexarm.
        """
        self.current_state = "initialize_rexarm"

        if not self.rexarm.initialize():
            print('Failed to initialize the rexarm')
            self.status_message = "State: Failed to initialize the rexarm!"
            time.sleep(5)
        self.next_state = "idle"
