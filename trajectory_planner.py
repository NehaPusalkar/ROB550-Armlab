"""!
! Trajectory planner.

TODO: build a trajectory generator and waypoint planner so it allows your state machine to iterate through the plan at
the desired command update rate.
"""
import numpy as np
import time

class TrajectoryPlanner():
    """!
    @brief      This class describes a trajectory planner.
    """

    def __init__(self, rexarm):
        """!
        @brief      Constructs a new instance.

        @param      rexarm  The rexarm
        """
        self.idle = True
        self.rexarm = rexarm
        self.initial_wp = None
        self.final_wp = None
        self.dt = 0.05 # command rate

    def set_initial_wp(self):
        """!
        @brief      TODO: Sets the initial wp to the current position.
        """
        self.initial_wp = self.rexarm.position_fb
        pass

    def set_final_wp(self, waypoint):
        """!
        @brief      TODO: Sets the final wp.

        @param      waypoint  The waypoint
        """
        self.final_wp = waypoint
        pass

    def go(self, max_speed=2.5):
        """!
        @brief      TODO Plan and execute the trajectory.

        @param      max_speed  The maximum speed
        """
        T = self.calc_time_from_waypoints(self.initial_wp, self.final_wp, max_speed)
        plan = self.generate_cubic_spline(self.initial_wp, self.final_wp, T)
        self.execute_plan(plan)
        pass

    def stop(self):
        """!
        @brief      TODO Stop the trajectory planner
        """

        pass

    def generate_quintic_spline(self, initial_wp, final_wp, max_speed):
        """!
        @brief      Generate the plan to get from initial to final waypoint.

        @param      initial_wp  The initial wp
        @param      final_wp    The final wp
        @param      max_speed   The maximum speed

        @return     The plan as num_steps x num_joints np.array
        """
        #Calculate Tf first
        #Since cons1 = np.array([[1,1,1],[3,4,5],[6,12,20]])
        cons1_inv = np.array([[10,-4,0.5], [-15,7,-1],[6,-3,0.5]])
        dq_max = max(abs(np.array(final_wp) - np.array(initial_wp)))
        res = np.dot(cons1_inv, np.array([[dq_max],[0],[0]]))
        cons2 = np.array([0.75, 0.5, 5/16])
        T = np.dot(cons2, res) / max_speed

        #Generate quintic spline
        H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
            [1,T,T**2,T**3,T**4,T**5],[0,1,2*T,3*T**2,4*T**3,5*T**4],[0,0,1,6*T,12*T**2,20*T**3]])
        t = np.arange(0,T,self.dt)
        H_inv = np.linalg.inv(H)
        output = []
        for i in range(4):
            B = np.array([initial_wp[i],0,0,final_wp[i],0,0])
            A = np.dot(H_inv, B)
            t_matrix = np.array([np.ones(len(t)), t, t**2, t**3, t**4, t**5])
            output.append(np.dot(np.transpose(t_matrix), A))
        output = np.transpose(np.array(output))
        return output



    def calc_time_from_waypoints(self, initial_wp, final_wp, max_speed):
        """!
        @brief      TODO Calculate the time to get from initial to final waypoint.

        @param      initial_wp  The initial wp
        @param      final_wp    The final wp
        @param      max_speed   The maximum speed

        @return     The amount of time to get to the final waypoint.
        """
        T = abs(np.array(final_wp) - np.array(initial_wp))/max_speed
        return max(T)

    def generate_cubic_spline(self, initial_wp, final_wp, T):
        """!
        @brief      TODO generate a cubic spline

        @param      initial_wp  The initial wp
        @param      final_wp    The final wp
        @param      T           Amount of time to get from initial to final waypoint

        @return     The plan as num_steps x num_joints np.array
        """
        H = np.array([[1,0,0,0],[0,1,0,0],[1,T,T**2,T**3],[0,1,2*T,3*T**2]])
        t = np.arange(0,T,self.dt)
        H_inv = np.linalg.inv(H)
        output = []
        for i in range(4):
            B = np.array([initial_wp[i],0,final_wp[i],0])
            A = np.dot(H_inv, B)
            t_matrix = np.array([np.ones(len(t)), t, t**2, t**3])
            output.append(np.dot(np.transpose(t_matrix), A))
        output = np.transpose(np.array(output))
        return output

    def execute_plan(self, plan, look_ahead=8):
        """!
        @brief      TODO: Execute the planed trajectory.

        @param      plan        The plan
        @param      look_ahead  The look ahead
        """
        print("---------")
        for i in range(len(plan)):
            self.rexarm.set_positions(plan[i])
            time.sleep(self.dt)
        pass
