import gym
import rospy
import roslaunch
import time
import numpy as np
import random
from squaternion import Quaternion
from math import pow, atan2, sqrt, exp, pi
from statistics import mean

from gym import utils, spaces
from gym_gazebo.project_envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, ModelStates
from turtlesim.msg import Pose
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
from gym.utils import seeding



class ProjectAcEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboProject.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.update_pose)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.action_time = 0.00000
        self.pose = Pose()
        self.goalpose = Pose()
        self.beforepose = Pose()
        self.startpose = Pose()

        self.goal = False
        self.goal_radius = 1.500
        self.update_subgoal = False
        self.robot_id = 2

        self.goalpose.x = 0.000
        self.goalpose.y = 0.000
        """
        self.goalx = [9.000,9.000,9.000,0.000,-9.000]
        self.goaly = [-9.000,9.000,-9.000,0.000,-9.000]
        """
        self.goalx = [0.000,8.500,8.500,-8.500,-8.500]
        self.goaly = [0.000,8.500,-8.500,8.500,-8.500]
        self.goalid = 0
        self.goalsum = 0
        
        self.get_pose(self.beforepose)
        self.subgoal_as_dist_to_goal = 30 # max. lidar's value
        self.target_angle = 0
        self.lidar_avg = 0
        self.start_mode = None

		# Action space design
        """
        self.action_space = spaces.Box(
            low = [min. linear velocity x, min. linear velocity y, min. angular velocity],
            high = [max. linear velocity x, max. linear velocity y, max. angular velocity],
            dtype = np.float32
        )
        """
        
        self.action_space = spaces.Box(
            low = np.array([-0.5, -0.5, -0.3]),
            high = np.array([0.5, 0.5, 0.3]),
            dtype = np.float32
        )
        """
        shape(lidar sensors + distance + angle,)
        """
        self.observation_space = spaces.Box(low = -1, high = 1, shape=(105,), dtype=np.float32)
		
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_start_mode(self, mode):
        # mode = "random" or "static"
        self.start_mode = mode

    def reset_vel(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
    
    def set_goal(self):
        self.goalpose.x = self.goalx[self.goalid]
        self.goalpose.y = self.goaly[self.goalid]

    # Set postion of the robot randomly
    def random_start(self):
        state_msg = ModelState()
        state_msg.model_name = 'robot'
        state_msg.pose.position.x = -6.0
        state_msg.pose.position.y = random.uniform(-1,1)
        state_msg.pose.position.z = 0.1
        quaternion = Quaternion.from_euler(0,0,random.uniform(-pi,pi))
        state_msg.pose.orientation.z = quaternion[3]
        state_msg.pose.orientation.w = quaternion[0]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except (rospy.ServiceException) as e:
            print("Service call failed: %s" %e)
    
    def static_start(self):
        state_msg = ModelState()
        state_msg.model_name = 'robot'
        quaternion = Quaternion.from_euler(0,0,0)
        state_msg.pose.position.x = -9.0
        state_msg.pose.position.y = -9.0
        state_msg.pose.orientation.z = quaternion[3]
        state_msg.pose.orientation.w = quaternion[0]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except (rospy.ServiceException) as e:
            print("Service call failed: %s" %e)

    def random_obstacle(self):
        obstacle_mode = 0#random.randint(0,1)
        
        for n in range(0,11):
            max_obs_acc = 1
            state_msg = ModelState()
            state_msg.model_name = 'obstacle_'+str(n)
            
            if n==0:
                state_msg.pose.position.x = random.uniform(7.3,9.7)    
                state_msg.pose.position.y = random.uniform(1.8,6.7)
            elif n==1:
                state_msg.pose.position.x = random.uniform(7.3,9.7)       
                state_msg.pose.position.y = random.uniform(-1.8,-6.7)
            elif n==2:
                state_msg.pose.position.x = random.uniform(3.3,5.7)       
                state_msg.pose.position.y = random.uniform(1.8,6.7)
            elif n==3:
                state_msg.pose.position.x = random.uniform(3.3,5.7)        
                state_msg.pose.position.y = random.uniform(-1.8,-6.7)
            elif n==4:
                state_msg.pose.position.x = random.uniform(-1.2,1.2)      
                state_msg.pose.position.y = random.uniform(1.8,6.7)
            elif n==5:
                state_msg.pose.position.x = random.uniform(-1.2,1.2)      
                state_msg.pose.position.y = random.uniform(-1.8,-6.7)
            elif n==6:
                state_msg.pose.position.x = random.uniform(-3.3,-5.7)      
                state_msg.pose.position.y = random.uniform(1.8,6.7)
            elif n==7:
                state_msg.pose.position.x = random.uniform(-3.3,-5.7)      
                state_msg.pose.position.y = random.uniform(-1.8,-6.7)
            elif n==8:
                state_msg.pose.position.x = random.uniform(-7.3,-9.7)      
                state_msg.pose.position.y = random.uniform(1.8,6.7)
            elif n==9:
                state_msg.pose.position.x = random.uniform(-7.3,-9.7)      
                state_msg.pose.position.y = random.uniform(-1.8,-6.7)
            elif n==10:
                state_msg.pose.position.x = random.uniform(-9.7,9.7)    
                state_msg.pose.position.y = random.uniform(-1.2,1.2)
                
            state_msg.twist.linear.y = 0.0
            if n == 10:
                state_msg.twist.linear.x = random.uniform(-max_obs_acc,max_obs_acc)
            else:
                state_msg.twist.linear.x = 0.0
                
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(state_msg)
            except (rospy.ServiceException) as e:
                print("Service call failed: %s" %e)
                
    def sim_time(self, data):
        self.sim_time = data

    def get_pose(self, vector):
        vector.x = self.pose.x
        vector.y = self.pose.y

    def update_pose(self, data):
        try:
            self.pose.x = data.pose[self.robot_id].position.x
            self.pose.y = data.pose[self.robot_id].position.y
            quaternion = Quaternion(data.pose[self.robot_id].orientation.x, data.pose[self.robot_id].orientation.y, data.pose[self.robot_id].orientation.z, data.pose[self.robot_id].orientation.w)
            self.pose.theta = (quaternion.to_euler()[0]/pi)+0.25
        except IndexError:
            None

    def euclidean_distance(self, now_pose, goal_pose):
        """Euclidean distance between current pose and the goal"""
        return sqrt(pow((goal_pose.x - now_pose.x), 2) + pow((goal_pose.y - now_pose.y), 2))

    def euclidean_angle(self, now_pose, goal_pose):
        #angle = [-pi,pi] = [-1,1]
        pi_rad = np.arcsin((goal_pose.y - now_pose.y)/self.euclidean_distance(now_pose, goal_pose))
        if (goal_pose.x - now_pose.x) < 0:
            if pi_rad >= 0:
                pi_rad = pi - pi_rad
            else:
                pi_rad += -pi
        return pi_rad/pi
    
    def calculate_target(self):
        cur_distance = min(self.euclidean_distance(self.pose, self.goalpose),30)
        body_to_target_angle = self.euclidean_angle(self.pose, self.goalpose)
        
        direction_to_target_angle = body_to_target_angle - self.pose.theta
        if (direction_to_target_angle > 1):
            direction_to_target_angle += -2
        elif (direction_to_target_angle < -1):
            direction_to_target_angle += 2
        elif direction_to_target_angle == 1 or direction_to_target_angle == -1:
            direction_to_target_angle = 1
        
        cur_distance = self.euclidean_distance(self.pose, self.goalpose)
        if cur_distance < self.subgoal_as_dist_to_goal :
            self.subgoal_as_dist_to_goal = cur_distance
            self.update_subgoal = True
        
        self.target_angle = direction_to_target_angle  
        
        return [cur_distance/30, self.target_angle]
    
    def calculate_observation(self,data):
        min_range = 0.300
        done = False
        state_list = []
        for i, item in enumerate(data[:-2]):
            if not np.isinf(data[i]):    
                if (min_range >= data[i] > 0):
                    done = True
                state_list += [data[i]/30]
            else:
                state_list += [1.0]
        
        state_list += data[-2:]
        self.lidar_avg = mean(state_list[:-2])*30
        state_tuple = tuple(state_list)
        return state_tuple, done

    def _step(self,action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
            self.get_pose(self.startpose)
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.linear.y = action[1]
        vel_cmd.angular.z = action[2]
    
        self.vel_pub.publish(vel_cmd)
        time.sleep(0.02)
        self.reset_vel()
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
        
        target_data = self.calculate_target()
        
        return [data.ranges, target_data] 
        
    def step(self, action):
        lidar_data = []
        target_data = []
        
        _data = self._step(action)
        lidar_data += list(_data[0])
        target_data += _data[1]
        data = lidar_data + target_data
        data = tuple(data)

        state, done = self.calculate_observation(data)
        
        cur_distance = self.euclidean_distance(self.pose, self.goalpose)
        distance = self.euclidean_distance(self.beforepose, self.goalpose) - cur_distance
        
        if cur_distance <= self.goal_radius: 
            self.goal = True
            self.goalsum += 1
            if self.goalid<4: self.goalid += 1
            else: self.goalid = 0
            self.set_goal()
            print("goal:",self.goalpose.x,self.goalpose.y)
        else: self.goal = False
        
        reward = distance + distance*exp(-abs(self.target_angle)/0.35) + 0.3*(1-exp((0.3-self.lidar_avg)/0.3)) + int(self.goal) - int(done)
        #print("d:"+str(round(prev_distance-cur_distance,4))+"|ang:"+str(round(self.target_angle,3))+"|R:"+str(round(reward,3)))
        
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
        
        #self.random_obstacle()
        self.goalid = 0
        self.goalsum = 0
        self.set_goal()
        
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if self.start_mode == 'random': self.random_start()
        elif self.start_mode == 'static': self.static_start()
        time.sleep(1)
        
        #read laser data
        _lidar_data = None
        while _lidar_data is None:
            try:
                _lidar_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
            
        lidar_data = []
        target_data = []   
        _target_data = self.calculate_target()
        for i in range(1):
            lidar_data += list(_lidar_data.ranges)
            target_data += _target_data
            data = lidar_data + target_data
        data = tuple(data)
            
        state, done = self.calculate_observation(data)

        self.beforepose.x = self.pose.x
        self.beforepose.y = self.pose.y
        self.subgoal_as_dist_to_goal = self.euclidean_distance(self.pose, self.goalpose)

        return state

