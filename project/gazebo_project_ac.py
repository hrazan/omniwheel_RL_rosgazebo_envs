import gym
import rospy
import roslaunch
import time
import numpy as np
import random

from math import pow, atan2, sqrt, exp

from gym import utils, spaces
from gym_gazebo.project_envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelState, ModelStates
from turtlesim.msg import Pose
from rosgraph_msgs.msg import Clock

from gazebo_msgs.srv import SetModelState

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class ProjectAcEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboProject.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.update_pose)
        rospy.Subscriber('/clock', Clock, self.sim_time)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.sim_time = Clock()
        self.action_time = 0.00000
        self.pose = Pose()
        self.goalpose = Pose()
        self.beforepose = Pose()
        self.startpose = Pose()

        self.before_avg_data = 0

        self.goal = False

        self.goalpose.x = 15.000
        self.goalpose.y = 0.000
        self.get_pose(self.beforepose)
        self.subgoal_as_dist_to_goal = 20 # max. lidar's value

		# Action space design
        """
        self.action_space = spaces.Box(
            low = [min. linear velocity x, min. linear velocity y, min. angular velocity],
            high = [max. linear velocity x, max. linear velocity y, max. angular velocity],
            dtype = np.float32
        )
        """
        self.action_space = spaces.Box(
            low = np.array([0.0, -0.3, -0.3]),
            high = np.array([0.3, 0.3, 0.3]),
            dtype = np.float32
        )
		
        self.observation_space = spaces.Box(low = 0, high = 20, shape=(109,), dtype=np.float32)
		
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_vel(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

    # Set postion of the robot randomly
    def random_start(self):
        state_msg = ModelState()
        state_msg.model_name = 'robot'
        state_msg.pose.position.x = 0
        state_msg.pose.position.y = random.uniform(-1,1)
        state_msg.pose.position.z = 0.1

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException, e:
            print("Service call failed: %s" %e)

    def random_obstacle(self):
        for n in range(0,10):
            state_msg = ModelState()
            state_msg.model_name = 'obstacle_'+str(n)
            #state_msg.pose.position.x = random.randint(1,9)
            if n<5:
                state_msg.pose.position.x = n+1
            else:
                state_msg.pose.position.x = 4-n
            state_msg.pose.position.y = random.uniform(-1,1)
            state_msg.twist.linear.x = 0.0
            state_msg.twist.linear.y = 0.0
            #state_msg.twist.linear.x = random.uniform(-0.3,0.3)
            #state_msg.twist.linear.y = random.uniform(-0.3,0.3)
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(state_msg)
            except rospy.ServiceException, e:
                print("Service call failed: %s" %e)

    def sim_time(self, data):
        self.sim_time = data

    def get_pose(self, vector):
        vector.x = self.pose.x
        vector.y = self.pose.y

    def update_pose(self, data):
        try:
            self.pose.x = round(data.pose[12].position.x, 4)
            self.pose.y = round(data.pose[12].position.y, 4)
        except IndexError:
            None

    def euclidean_distance(self, now_pose, goal_pose):
        """Euclidean distance between current pose and the goal"""
        return round(sqrt(pow((goal_pose.x - now_pose.x), 2) + pow((goal_pose.y - now_pose.y), 2)),4)

    def calculate_observation(self,data):
        min_range = 0.301
        done = False
        state_list = []
        for i, item in enumerate(data.ranges):
            if not np.isinf(data.ranges[i]):    
                if (min_range > data.ranges[i] > 0):
                    done = True
                state_list += [data.ranges[i]]
            else:
                state_list += [20]
                #print('ERROR: INFINITY!')
        distance = self.euclidean_distance(self.pose, self.goalpose)
        state_list += [distance]
        state_tuple = tuple(state_list)
        return state_tuple, done

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
            self.get_pose(self.startpose)
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.goal = False

        #start = self.sim_time
        #print(action)
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.linear.y = action[1]
        vel_cmd.angular.z = action[2]
        self.vel_pub.publish(vel_cmd)
        #print(action[0][0],action[0][1],action[0][2])

        time.sleep(0.01)
        self.reset_vel()
        #self.action_time = ((self.sim_time.clock.secs - start.clock.secs)*1000000000) + (self.sim_time.clock.nsecs - start.clock.nsecs)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.calculate_observation(data)
        avg_data = sum(state)/len(state)

        distance = self.euclidean_distance(self.pose, self.goalpose)

        if not done:
            if  distance < self.euclidean_distance(self.beforepose, self.goalpose) and distance < self.subgoal_as_dist_to_goal :
                reward = 100/distance
                self.subgoal_as_dist_to_goal = distance
            else: reward = 0.1
        else:
            reward = 0.01
            
        #reward = np.array(reward).reshape((1,))

        self.beforepose.x = self.pose.x
        self.beforepose.y = self.pose.y
        
        return np.asarray(state), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        
        #time.sleep(0.5)
        #time.sleep(1)
        self.random_obstacle()
        
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        time.sleep(1)
        #self.random_start()
        
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
            
        state, done = self.calculate_observation(data)

        self.beforepose.x = self.pose.x
        self.beforepose.y = self.pose.y
        self.subgoal_as_dist_to_goal = 1000

        return np.asarray(state)
