import gym
import rospy
import roslaunch
import time
import random
import numpy as np

from math import pow, atan2, sqrt, exp
from statistics import mean

from gym import utils, spaces
from gym_gazebo.project_envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from turtlesim.msg import Pose
from rosgraph_msgs.msg import Clock

from gazebo_msgs.srv import SetModelState

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class ProjectEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboProject.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        #self.before_pose = rospy.Publisher('/gazebo/before_model_states', Pose, queue_size=5)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.update_pose)
        rospy.Subscriber('/clock', Clock, self.sim_time)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(4) #F,L,R,B
        self.reward_range = (-np.inf, np.inf)	
        
        self.sim_time = Clock()
        self.action_time = 0.00000
        self.pose = Pose()
        self.goalpose = Pose()
        self.beforepose = Pose()
        self.startpose = Pose()

        self._seed()

        self.goal = False
        self.randomstart = False

        self.goalpose.x = 2.000 # map0: 1.5, map1: 1.25, map2: 2.0
        self.goalpose.y = -0.250 # map0,1: 0.0, map2: -0.25
        self.get_pose(self.beforepose)
        #self.beforepose.x = 0.0000
        #self.beforepose.y = 0.0000

    def set_randomstart(self, set_bool):
        self.randomstart = set_bool
    
    # Set postion of the robot randomly
    def random_start(self):
        state_msg = ModelState()
        state_msg.model_name = 'robot'
        # MAP 0
        """
        area = random.randint(1,3)
        if area == 1:
            state_msg.pose.position.x = random.uniform(-0.4,0.4)
            state_msg.pose.position.y = random.uniform(-1.15,0.35)
        elif area == 2:
            state_msg.pose.position.x = random.uniform(-0.4,1.9)
            state_msg.pose.position.y = random.uniform(0.35,1.15)
        else:
            state_msg.pose.position.x = random.uniform(1.1,1.9)
            state_msg.pose.position.y = random.uniform(-1.15,0.35)
        """ 
        
        # MAP 2
        area = random.randint(1,6)
        if area == 1:
            state_msg.pose.position.x = random.uniform(-0.15,0.15)
            state_msg.pose.position.y = random.uniform(-1.15,1.15)
        elif area == 2:
            state_msg.pose.position.x = random.uniform(0.15,0.85)
            state_msg.pose.position.y = random.uniform(0.35,0.65)
        elif area == 3:
            state_msg.pose.position.x = random.uniform(0.85,1.15)
            state_msg.pose.position.y = random.uniform(-1.15,1.15)
        elif area == 4:
            state_msg.pose.position.x = random.uniform(1.15,1.85)
            state_msg.pose.position.y = random.uniform(-1.15,-0.85)
        elif area == 5:
            state_msg.pose.position.x = random.uniform(1.15,1.85)
            state_msg.pose.position.y = random.uniform(0.85,1.15)
        else:
            state_msg.pose.position.x = random.uniform(1.85,2.15)
            state_msg.pose.position.y = random.uniform(-1.15,1.15)
        

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except (rospy.ServiceException) as e:
            print("Service call failed: %s" %e)

    def reset_vel(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

    def sim_time(self, data):
        self.sim_time = data
        #nsecs = data.clock.nsecs/1000000000
        #self.sim_time = data.clock.secs + nsecs

    def get_pose(self, vector):
        vector.x = self.pose.x
        vector.y = self.pose.y

    def update_pose(self, data):
        try:
            self.pose.x = round(data.pose[2].position.x, 4)
            self.pose.y = round(data.pose[2].position.y, 4)
        except IndexError:
            #print("Index Error! Wait a momment")
            self.pose.x = 0
            self.pose.y = 0
        #self.pose.pose.position.x = pose[2].position.x 
        #x = data.pose.pose.position.x
        #y = data.pose.pose.position.y
        #print('data.pose[1]')
        #print(data.pose[2].position)
        #print('data.pose[2]')
        #print(data.pose[2])
        #print("Robot Pose:")
        #print("x = ", x)
        #print("y = ", y)
        #print(data)

    def euclidean_distance(self, now_pose, goal_pose):
        """Euclidean distance between current pose and the goal"""
        return round(sqrt(pow((goal_pose.x - now_pose.x), 2) + pow((goal_pose.y - now_pose.y), 2)),4)

    def dist_to_16digit(self, state_data):            
        if (state_data<=0.25): state_data_char = '0'
        elif (0.25<state_data<=0.5): state_data_char = '1'
        elif (0.5<state_data<=0.75): state_data_char = '2'
        elif (0.75<state_data<=1): state_data_char = '3'
        elif (1<state_data<=1.25): state_data_char = '4'
        elif (1.25<state_data<=1.5): state_data_char = '5'
        elif (1.5<state_data<=1.75): state_data_char = '6'
        elif (1.75<state_data<=2): state_data_char = '7'
        elif (2<state_data<=2.25): state_data_char = '8'                
        elif (2.25<state_data<=2.5): state_data_char = '9'
        elif (2.5<state_data<=2.75): state_data_char = 'A'
        elif (2.75<state_data<=3): state_data_char = 'B'
        elif (3<state_data<=3.25): state_data_char = 'C'
        elif (3.25<state_data<=3.5): state_data_char = 'D'
        elif (3.5<state_data<=3.75): state_data_char = 'E'
        elif (3.75<state_data): state_data_char = 'F'
        
        return state_data_char      
    
    def discretize_observation(self,data,new_ranges):
        state_1, state_2, state_3, state_4, state_5 = [],[],[],[],[]
        min_range = 0.301
        done = False
        #16_state_data = ""
        #print(data.ranges)
        for i, item in enumerate(data.ranges):
            if item == float ('Inf') or np.isinf(item):
                state_data = 4.25
            elif np.isnan(item):
                state_data = 0
            else:
                state_data = item

            if 0<=i<=216:
                state_1.append(state_data)
                if len(state_1)==217:
                    state_1 = self.dist_to_16digit(mean(state_1))
            if 216<=i<=432:
                state_2.append(state_data)
                if len(state_2)==217:
                    state_2 = self.dist_to_16digit(mean(state_2))
            if 432<=i<=648:
                state_3.append(state_data)
                if len(state_3)==217:
                    state_3 = self.dist_to_16digit(mean(state_3))
            if 648<=i<=864:
                state_4.append(state_data)
                if len(state_4)==217:
                    state_4 = self.dist_to_16digit(mean(state_4))
            if 864<=i<=1080:
                state_5.append(state_data)
                if len(state_5)==217:
                    state_5 = self.dist_to_16digit(mean(state_5))

            if (min_range > state_data > 0):
                done = True
        
        states = state_1+state_2+state_3+state_4+state_5

        #print(done)
        #print(discretized_ranges)
        
        return states, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print('3:',self.pose)
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #print('4:',self.pose)
            self.unpause()
            self.get_pose(self.startpose)
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.goal = False

        start = self.sim_time

        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.5
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = -0.5
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 3: #BACK
            vel_cmd = Twist()
            vel_cmd.linear.x = -0.5
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)

        time.sleep(0.02)
        self.reset_vel()
        self.action_time = ((self.sim_time.clock.secs - start.clock.secs)*1000000000) + (self.sim_time.clock.nsecs - start.clock.nsecs)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                #print(data)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.discretize_observation(data,5)

        distance = self.euclidean_distance(self.pose, self.goalpose)

        if not done:
            if self.euclidean_distance(self.beforepose, self.goalpose) >= distance:
                if distance <= 0.25:
                    reward = 0
                    done = True
                    self.goal = True
                    print("GOOOOOOOAAAAAAAAAAL!")
                    #print("Simulation time (secs): "+str(self.sim_time))
                else:
                    reward = -1
            else:
                reward = -10
        else:
            reward = -1000000

        #print("Action:", action)
        #print("Distance to Goal(before):", self.euclidean_distance(self.beforepose, self.goalpose))
        #print("Distance to Goal:", self.euclidean_distance(self.pose, self.goalpose))
        #print("Reward:", reward)

        self.beforepose.x = self.pose.x
        self.beforepose.y = self.pose.y

        return state, reward, done, {}

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if self.randomstart: self.random_start()
        time.sleep(1.5) #DQN25x=3,Q25x=1.5

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.discretize_observation(data,5)
        #print(state)

        return state
