#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import json

import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings("ignore")
import time

# custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ergodic_control import models, utilities

# ROS
import rospy
import rospkg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import OccupancyGrid
from tf import TransformListener

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandTOL, CommandTOLRequest



class ErgodicNode():
    def __init__(self):
        rospy.init_node("ergodic_node", anonymous=True)
        pkg_path = rospkg.RosPack().get_path('ergodic_control')
        """
        ===============================
        Load map
        ===============================
        """
        map_name = 'simpleMap_05'
        map_path = os.path.join(pkg_path,  'example_maps/', map_name + '.npy')
        self.map = np.load(map_path)
        free_cells = np.array(np.where(self.map == 0)).T
        
        # Is the map closed?
        self.closed_map = True
        # Extend the map with 1 cell to avoid index out of bounds
        padded_map = np.pad(self.map, 1, 'constant', constant_values=1)
        self.occ_map = utilities.get_occupied_polygon(padded_map)
        self.occ_map = self.occ_map - 1


        x_min, x_max = 0, self.map.shape[0]
        y_min, y_max = 0, self.map.shape[1]
        self.grid_x, self.grid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
        self.grid = np.vstack([self.grid_x.flatten(), self.grid_y.flatten()]).T

        """
        ===============================
        Parameters
        ===============================
        """
        param_file = os.path.join(pkg_path, 'params/', 'doubleintegrator.json')

        with open(param_file, 'r') as f:
            param_data = json.load(f)

        self.param = lambda: None

        for key, value in param_data.items():
            setattr(self.param, key, value)

        np.random.seed(self.param.random_seed)

        # param.max_dtheta = np.pi / param.max_dtheta # Maximum angular velocity (rad/s)
        self.param.nbResX = self.map.shape[0] # Number of cells in the x-direction
        self.param.nbResY = self.map.shape[1] # Number of cells in the y-direction

        # CFL (Courant-Friedrichs-Lewy) condition for implicit integration and heat equation stability
        self.param.dt = min(
            1.0, (self.param.dx ** 2) / (4.0 * np.max(self.param.alpha))
        )


        self.agents = [None] * self.param.nbAgents

        self.spatial_decay = 1000
        self.temporal_decay = 1000
        self.min_safe_range = 1.0  # Minimum safe distance between agents
        

        """
        ===============================
        ROS Parameters
        ===============================
        """

        # ------------------------------------------------
        # self.robot_id : Name of the robot, e.g. UAV1 --> 1
        # self.id : Index of the robot in the list of agents, e.g. UAV1 --> 0
        # ------------------------------------------------
        self.robot_id = rospy.get_param("~robot_id", 0)
        self.altitude = rospy.get_param("~takeoff_altitude", 10.0)
        self.takeoff_time = rospy.get_param("~takeoff_time", 10.0)
        self.save_video = rospy.get_param("~save_video", False)
        env_id = os.getenv("UAV_ID")
        if env_id is not None:
            try:
                self.robot_id = int(env_id)
                print(f"Found env variable UAV_ID={self.robot_id}")
            except ValueError:
                print("Environment variable UAV_ID is not a valid integer.")

        sorted_names = sorted(self.param.ids)
        self.id = sorted_names.index(self.robot_id)
        print("I'm UAV", self.robot_id, "with id ", self.id)
        self.agent = self.agents[self.id]
        self.rate = rospy.Rate(self.param.dt)

        # Subscribers and Publishers
        for i in range(self.param.nbAgents):
            self.odom_subs = rospy.Subscriber(
                f"/supervisor/uav{self.param.ids[i]}/odom", 
                Odometry, 
                self.odom_callback, 
                callback_args=i
            )

        # mavros
        self.mavros_state = None
        self.state_sub = rospy.Subscriber(
            "mavros/state",
            State,
            self.state_cb,
        )
        self.arming_client = rospy.ServiceProxy(
            "mavros/cmd/arming",
            CommandBool
        )
        self.set_mode_client = rospy.ServiceProxy(
            "mavros/set_mode",
            SetMode
        )

        self.local_pos_pub = rospy.Publisher(
            "mavros/setpoint_position/local",
            PoseStamped,
            queue_size=10
        )

        self.landing_client = rospy.ServiceProxy(
            "mavros/cmd/land",
            CommandTOL
        )

        # occupancy map publisher
        self.map_pub = rospy.Publisher(
            "occupancy_grid", 
            OccupancyGrid, 
            queue_size=10
        )
        self.map_timer = rospy.Timer(
            rospy.Duration(1.0), 
            self.map_cb
        )

        self.goal_density_pub = rospy.Publisher(
            "goal_density",
            OccupancyGrid,
            queue_size=10
        )

        self.mean_pred_pub = rospy.Publisher(
            "mean_prediction",
            OccupancyGrid,
            queue_size=10
        )

        # self.map_msg = OccupancyGrid()
        # self.map_msg.header.frame_id = "map"
        # self.map_msg.info.resolution = 1
        # self.map_msg.info.width = self.map.shape[1]
        # self.map_msg.info.height = self.map.shape[0]
        # self.map_msg.info.origin.position.x = self.map.shape[1]
        # self.map_msg.info.origin.position.y = 0.0  # In ROS, the origin is at the bottom left corner
        # self.map_msg.info.origin.position.z = 0.0
        # # rotate 90 degrees
        # self.map_msg.info.origin.orientation.x = 0.0
        # self.map_msg.info.origin.orientation.y = 0.0
        # self.map_msg.info.origin.orientation.z = 0.707  # sin(45 degrees)
        # self.map_msg.info.origin.orientation.w = 0.707
        # self.map_msg.data = (np.flipud(self.map).flatten(order='C') * 100).astype(int).tolist() 

        # velocity publisher
        self.vel_pub = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", 
            TwistStamped, 
            queue_size=10
        )

        rospy.on_shutdown(self.shutdown_hook)

        """
        ===============================
        Goal Density
        ===============================
        """
        free_cells = np.array(np.where(self.map == 0)).T  # Replace with your actual free cell array
        _, density_map = utilities.generate_gmm_on_map(self.map,
                                                    free_cells,
                                                    self.param.nbGaussian,
                                                    self.param.nbParticles,
                                                    self.param.nbVar,
                                                    random_state=self.param.random_seed)
        # means = np.array([[40, 40], [20, 45], [8, 8]])
        # cov = np.array([[[20, 0], [0, 20]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
        # density_map = utilities.gauss_pdf(grid, means[0], cov[0]) #+ \
        #             # utilities.gauss_pdf(grid, means[1], cov[1])
        #                 # utilities.gauss_pdf(grid, means[2], cov[2])

        # norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
        norm_density_map = density_map.reshape(self.map.shape)

        # Compute the area of the map
        cell_area = self.param.dx * self.param.dx
        self.param.area = np.sum(self.map == 0) * cell_area

        self.goal_density = np.zeros_like(self.map)
        self.goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
        # Min-max normalize the goal density
        self.goal_density = np.abs(self.goal_density)
        self.goal_density = utilities.min_max_normalize(self.goal_density) # remember to normalize
        goal_density_norm = utilities.normalize_mat(self.goal_density)

        """
        ===============================
        Initialize heat equation related parameters
        ===============================
        """
        self.param.width = self.map.shape[0]
        self.param.height = self.map.shape[1]

        self.param.beta = self.param.beta / self.param.area # Eq. 17 - Beta normalized
        self.param.local_cooling = self.param.local_cooling / self.param.area # Eq. 16 - Local cooling normalized

        local_cooling = np.zeros_like(self.goal_density) # The local cooling
        coverage_density = np.zeros_like(self.goal_density) # The coverage density

        while (all(agent is None for agent in self.agents)):
            rospy.sleep(self.param.dt)

        for agent in self.agents:
            tmp = utilities.init_fov(self.param.fov_deg, self.param.fov_depth)
            agent.fov_edges = utilities.rotate_and_translate(tmp, agent.x, agent.theta)

        coverage_block = utilities.agent_block(self.param.nbVar, self.param.min_kernel_val, self.param.agent_radius)
        self.param.kernel_size = coverage_block.shape[0]

        """
        ================
        Gaussian process
        ================
        """
        noise = 0.005
        kernel = (
            C(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))  # space
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
        )

        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-5, normalize_y=False)

        self.gpr.kernel_ = kernel

        for agent in self.agents:
            agent.heat = np.empty_like(self.goal_density)
            agent.samples = np.empty((0, 4))
            agent.subset = np.empty((0, 4))
            agent.coverage_density_hist = np.zeros((50, 2, self.param.nbDataPoints), dtype=int)
            agent.coverage_density_prob_hist = np.zeros((50, self.param.nbDataPoints), dtype=float)

            agent.neighbors = []
            agent.last_neighbors = []
            agent.local_cooling = np.zeros_like(self.goal_density)
            agent.coverage_density = np.zeros_like(self.goal_density)

        self.adjacency_matrix = np.eye(self.param.nbAgents)

        self.step = 0

        for i in range(self.param.nbAgents):
            print("Agent", i, "initialized with id", self.agents[i].id)
            self.agents[i].heat = np.empty_like(self.goal_density)
            self.agents[i].samples = np.empty((0, 4))
            self.agents[i].subset = np.empty((0, 4))
            self.agents[i].coverage_density_hist = np.zeros((50, 2, self.param.nbDataPoints), dtype=int)
            self.agents[i].coverage_density_prob_hist = np.zeros((50, self.param.nbDataPoints), dtype=float)
            self.agents[i].neighbors = []
            self.agents[i].last_neighbors = []
            self.agents[i].local_cooling = np.zeros_like(self.goal_density)
            self.agents[i].coverage_density = np.zeros_like(self.goal_density)
        print("All agents initialized.")

        """
        ===============================
        MAVROS setup
        ===============================
        """
        while not self.mavros_state.connected:
            print("Waiting for MAVROS connection...")
            rospy.sleep(self.param.dt)
        print("MAVROS connected.")
        # print("Waiting 3 seconds...")
        # for _ in range(int(3/self.param.dt)):
        #     self.rate.sleep()
        
        self.tf_listener = TransformListener()
        self.start_pose = PoseStamped()
        self.takeoff_pose = PoseStamped()
        # Wait for the transform to be available
        self.tf_listener.waitForTransform(
            f"uav{self.robot_id}/local_origin", 
            f"uav{self.robot_id}/base_link", 
            rospy.Time(0), 
            rospy.Duration(4.0)
        )
        # Get the transform
        try:
            (pos, rot) = self.tf_listener.lookupTransform(
                f"uav{self.robot_id}/local_origin",
                f"uav{self.robot_id}/base_link",
                rospy.Time(0)
            )
            self.start_pose.header.frame_id = f"uav{self.robot_id}/local_origin"
            self.start_pose.pose.position.x = pos[0]
            self.start_pose.pose.position.y = pos[1]
            self.start_pose.pose.position.z = pos[2]
            print(f"Start pose: {self.start_pose.pose.position.x}, {self.start_pose.pose.position.y}, {self.start_pose.pose.position.z}")

            self.takeoff_pose.header.frame_id = f"uav{self.robot_id}/local_origin"
            self.takeoff_pose.pose.position.x = pos[0]
            self.takeoff_pose.pose.position.y = pos[1]
            self.takeoff_pose.pose.position.z = self.start_pose.pose.position.z + self.altitude
        except (rospy.ServiceException, rospy.ROSException) as e:
            print("Error getting transform:", e)
            rospy.signal_shutdown("Transform not available")
        
        # send a few setpoints
        for i in range(100):
            self.local_pos_pub.publish(self.start_pose)
            rospy.sleep(0.01)
        
        self.last_request = rospy.Time.now()


        
        # main loop timer
        self.timer = rospy.Timer(
            rospy.Duration(self.param.dt), 
            self.timer_cb
        )
        

        print("Initialization complete. Starting main loop...")

    def plot_results(self):
        # Plot mean and uncertainty of the GPR predictions
        # fig = plt.figure(figsize=(12, 5))
        # ax = fig.add_subplot(131)
        # ax.set_aspect('equal')
        # ax.contourf(self.grid_x, self.grid_y, self.goal_density, cmap='RdPu', levels=10)
        # ax.pcolormesh(self.grid_x, self.grid_y, np.where(self.map == 0, np.nan, self.map), cmap='gray')
        # ax.set_title('Ground Truth Goal Density')

        # ax = fig.add_subplot(132)
        # ax.set_aspect('equal')
        # ax.contourf(self.grid_x, self.grid_y, self.agents[0].mu.reshape(self.map.shape), cmap='RdPu', levels=10)
        # ax.pcolormesh(self.grid_x, self.grid_y, np.where(self.map == 0, np.nan, self.map), cmap='gray')
        # ax.set_title('GPR Mean Prediction')

        # ax = fig.add_subplot(133)
        # ax.set_aspect('equal')
        # ax.contourf(self.grid_x, self.grid_y, self.agents[0].std.reshape(self.map.shape), cmap='RdPu', levels=10)
        # ax.pcolormesh(self.grid_x, self.grid_y, np.where(map == 0, np.nan, self.map), cmap='gray')
        # ax.set_title('GPR Uncertainty (Std Dev)')

        # plt.show(block=True)

        # Create video of the agents moving
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        def update(frame):
            print(f"Updating frame {frame + 1}/{self.param.nbDataPoints}")
            ax.clear()
            ax.set_aspect('equal')
            ax.contourf(self.grid_x, self.grid_y, self.goal_density, cmap='RdPu', levels=10)
            ax.pcolormesh(self.grid_x, self.grid_y, np.where(self.map == 0, np.nan, self.map), cmap='gray')

            for agent in self.agents:
                # Plot the agents' paths
                ax.plot(agent.x_hist[:frame+1, 0], agent.x_hist[:frame+1, 1], color=f'C{agent.id}', alpha=0.5, lw=2)
                # Plot the agents' current positions
                ax.scatter(agent.x_hist[frame, 0], agent.x_hist[frame, 1], c=f'C{agent.id}', s=100, marker='x', label=f'Agent {agent.id} End')
                # Plot the heading
                ax.quiver(agent.x_hist[frame, 0], agent.x_hist[frame, 1], np.cos(agent.x_hist[frame, 2]), np.sin(agent.x_hist[frame, 2]), scale=3, scale_units='inches', color=f'C{agent.id}')
                # Draw the FOV
                # fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x_hist[frame], agent.fov_edges, occ_map, closed_map=True)
                # ax.fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color=f'C{agent.id}', alpha=0.3)

            ax.set_title(f'Frame {frame + 1}')

            return ax,
        

        ani = animation.FuncAnimation(fig, update, frames=np.arange(self.param.nbDataPoints, step=10), repeat=False)
        date = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        pkg_path = rospkg.RosPack().get_path('ergodic_control')
        filename = pkg_path+'/ros_simulation_' + date + '.mp4'
        print("Filename:", filename)
        ani.save(filename, writer='ffmpeg', fps=30)
        print("Saved video.")

    

    def shutdown_hook(self):
        rospy.loginfo("Shutting down Ergodic Node...")
        self.timer.shutdown()
        self.map_timer.shutdown()
        
        # Land the vehicle
        print(f"Landing UAV {self.robot_id}...")
        land_cmd = CommandTOLRequest()
        land_cmd.altitude = 0.0
        land_cmd.latitude = 0.0
        land_cmd.longitude = 0.0
        land_cmd.yaw = 0.0
        land_cmd.min_pitch = 0.0
        if self.landing_client.call(land_cmd).success:
            rospy.loginfo(f"UAV {self.robot_id} landed successfully.")
        
        if self.save_video:
            print("Plotting results...")
            self.plot_results()

        
        # self.local_pos_pub.unregister()
        # self.vel_pub.unregister()
        # self.state_sub.unregister()
        # for i in range(self.param.nbAgents):
        #     self.odom_subs[i].unregister()
        # self.map_pub.unregister()
        # self.arming_client.unregister()
        # self.set_mode_client.unregister()
        # self.landing_client.unregister()
        rospy.loginfo("Ergodic Node shutdown complete.")

    def state_cb(self, msg):
        self.mavros_state = msg


    def odom_callback(self, msg, i):
        # Initialize the agent if not already done
        if self.agents[i] is None:
            x0 = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            self.agents[i] = models.DoubleIntegratorAgent(
                x=x0,
                max_dx=self.param.max_dx,
                max_ddx=self.param.max_ddx,
                max_dtheta=self.param.max_dtheta,
                max_ddtheta=self.param.max_ddtheta,
                dt=self.param.dt,
                id=i
            )
            self.agents[i].x_hist = np.array([[x0[0], x0[1], 0.0]])  # Initialize history with position and zero orientation
            print("Initialized agent ", i)

        # Update position and orientation
        self.agents[i].x = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation = msg.pose.pose.orientation
        self.agents[i].theta = np.arctan2(
            2.0 * (orientation.z * orientation.w + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)
        )
        xyth = np.array([self.agents[i].x[0], self.agents[i].x[1], self.agents[i].theta])
        self.agents[i].x_hist = np.vstack((self.agents[i].x_hist, xyth))

    def map_cb(self, event):
        self.publish_map(self.map, self.map_pub)
        if hasattr(self, 'goal_density'):
            self.publish_map(self.goal_density, self.goal_density_pub)
        for agent in self.agents:
            if agent is not None and hasattr(agent, "mu"):
                map_data = agent.mu.reshape(self.map.shape)
                self.publish_map(map_data, self.mean_pred_pub)

    def publish_map(self, map_data, pub):
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = "map"
        map_msg.info.resolution = 1
        map_msg.info.width = map_data.shape[1]
        map_msg.info.height = map_data.shape[0]
        map_msg.info.origin.position.x = map_data.shape[0]
        map_msg.info.origin.position.y = 0.0 
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.x = 0.0
        map_msg.info.origin.orientation.y = 0.0
        map_msg.info.origin.orientation.z = 0.707
        map_msg.info.origin.orientation.w = 0.707
        map_msg.data = (np.flipud(map_data).flatten(order='C') * 100).astype(int).tolist() 
        pub.publish(map_msg)

    def timer_cb(self, event):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = "OFFBOARD"
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        if self.mavros_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_request > rospy.Duration(2.0)):
            if self.set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("Offboard enabled")
            self.last_request = rospy.Time.now()
        else:
            if not self.mavros_state.armed and (rospy.Time.now() - self.last_request > rospy.Duration(2.0)):
                if self.arming_client.call(arm_cmd).success:
                    rospy.loginfo("Vehicle armed")
                self.last_request = rospy.Time.now()

        if rospy.Time.now() - self.last_request < rospy.Duration(self.takeoff_time):
            self.takeoff_pose.header.frame_id = f"uav{self.robot_id}/local_origin"
            self.takeoff_pose.header.stamp = rospy.Time.now()
            self.local_pos_pub.publish(self.takeoff_pose)
            self.t_start = rospy.Time.now().to_sec()
        else:
            if self.step % 10 == 0:
                print(f"Step {self.step}")
            
            if self.param.nbAgents > 1 and self.step > 0:
                self.adjacency_matrix = utilities.share_samples(self.agents, self.map, self.param.sens_range, self.adjacency_matrix)

            self.agents[self.id].local_cooling = np.zeros_like(self.goal_density)

            tmp = utilities.init_fov(self.param.fov_deg, self.param.fov_depth)
            fov_edges_moved = utilities.rotate_and_translate(tmp, self.agents[self.id].x, self.agents[self.id].theta)
            fov_edges_clipped = utilities.clip_polygon_no_convex(self.agents[self.id].x, fov_edges_moved, self.occ_map, self.closed_map)
            fov_points = utilities.insidepolygon(fov_edges_clipped).astype(int)

            # Delete points outside the box
            fov_probs = utilities.fov_coverage_block(fov_points, fov_edges_clipped, self.param.fov_depth)

            self.agents[self.id].coverage_density[fov_points[:, 0], fov_points[:, 1]] += fov_probs

            self.agents[self.id].fov_edges = fov_edges_moved

            """ Goal density sampling """
            y = self.goal_density[fov_points[:, 0], fov_points[:, 1]]  # + np.random.normal(0, noise, len(fov_points))
            dataset = np.hstack((fov_points, time.time() * np.ones((fov_points.shape[0], 1), dtype=int).reshape(-1, 1), y.reshape(-1, 1)))

            self.agents[self.id].samples = np.vstack((self.agents[self.id].samples, dataset))
            self.agents[self.id].subset = self.agents[self.id].subset[np.argsort(self.agents[self.id].subset[:, 2])]

            # Mantovani et al. 2024 ======================================================================
            if self.step > 0:
                # Filter samples based on standard deviation threshold
                std_test = self.agents[self.id].std[self.agents[self.id].samples[:, 0].astype(int), self.agents[self.id].samples[:, 1].astype(int)]
                self.agents[self.id].samples = self.agents[self.id].samples[std_test > 0.75]

            # Only proceed if there are samples to process
            if len(self.agents[self.id].samples) != 0:
                pooled_samples = utilities.max_pooling(self.agents[self.id].samples, 5)
                self.agents[self.id].subset = np.unique(np.vstack((self.agents[self.id].subset, pooled_samples)), axis=0)

                if self.step > 0:
                    self.gpr.fit(self.agents[self.id].subset[:, :2], self.agents[self.id].subset[:, 3])

            # Compute decay matrices
            D, d = utilities.compute_spatio_decay_matrix(self.agents[self.id].subset[:, :2], self.spatial_decay, self.agents[self.id].x)
            T, t = utilities.compute_temporal_decay_matrix(self.agents[self.id].subset[:, 2], time.time(), self.temporal_decay)

            # Compute combo density and update agent state
            self.agents[self.id].combo_density, self.agents[self.id].mu, self.agents[self.id].std = utilities.compute_combo(
                self.agents[self.id].subset[:, :2],
                self.agents[self.id].subset[:, 3],
                self.grid,
                self.map,
                self.gpr.kernel_,
                D,
                T,
                d,
                t
            )

            # Clear low standard deviation values
            # agent.std[agent.std < 0.3] = 0

            print(f"Agent {self.agents[self.id].id} subset: {len(self.agents[self.id].subset)}")
            # Pratissoli et al. 2025 =====================================================================
            if self.step > 0:
                # Keep only the samples with uncertainty low enough
                std_test = self.agents[self.id].std[self.agents[self.id].subset[:, 0].astype(int), self.agents[self.id].subset[:, 1].astype(int)]
                self.agents[self.id].subset = self.agents[self.id].subset[std_test < 0.75]
                self.agents[self.id].subset = self.agents[self.id].subset[np.argsort(self.agents[self.id].subset[:, 2])]


            if self.step == 0:
                self.agents[self.id].heat = np.array(utilities.normalize_mat(self.agents[self.id].combo_density))

            diff = utilities.normalize_mat(self.agents[self.id].combo_density) - utilities.normalize_mat(self.agents[self.id].coverage_density)

            source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
            source = np.where(self.map == 0, source, 0)
            self.agents[self.id].source = utilities.normalize_mat(source) * self.param.area # Eq. 14 - Source term scaled

            # ergodic_metric[step, agent.id] = np.linalg.norm(agent.source) * param.dt # Eq. 15 - Ergodic metric

            current_heat = utilities.update_heat_optimized(
                self.agents[self.id].heat,
                self.agents[self.id].source,
                self.map,
                self.agents[self.id].local_cooling,
                self.param.dt,
                self.param.alpha,
                self.param.source_strength,
                self.param.beta,
                self.param.local_cooling,
                self.param.dx
            )

            self.agents[self.id].heat = current_heat.astype(np.float32)

            gradient_y, gradient_x = np.gradient(self.agents[self.id].heat.T, 1, 1)

            gradient_x /= np.linalg.norm(gradient_x) + 1e-6
            gradient_y /= np.linalg.norm(gradient_y) + 1e-6

            # Update the agent
            self.agents[self.id].grad = utilities.calculate_gradient_map(
                self.param, self.agents[self.id], gradient_x, gradient_y, self.map
            )

            if len(self.agents[self.id].neighbors) > 0:
                for neighbor in self.agents[self.id].neighbors:
                    neighbor_agent = self.agents[neighbor]
                    q_ij = np.linalg.norm(self.agents[self.id].x - neighbor_agent.x)**2
                    p = 4 * (self.param.sens_range**2 - self.min_safe_range**2) * (q_ij - self.param.sens_range**2) * (agent.x - neighbor_agent.x) / \
                        (q_ij - self.min_safe_range**2)**3
                    # Control law, we want to move away from the neighbor
                    self.agents[self.id].grad -= p

            # v_target and theta_target
            k_target = 1
            v_target = k_target * np.array([self.agents[self.id].grad[0], self.agents[self.id].grad[1]])

            theta_target = np.arctan2(self.agents[self.id].grad[1], self.agents[self.id].grad[0])

            u = self.agents[self.id].get_acceleration(v_target, theta_target, penalize_lateral=True)
            self.agents[self.id].update(u)

            # get linear and angular velocities
            vel = self.agents[self.id].v
            omega = self.agents[self.id].omega

            # convert to ROS message
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.twist.linear.x = vel[0]
            msg.twist.linear.y = vel[1]
            msg.twist.linear.z = 0.0
            msg.twist.angular.z = omega
            self.vel_pub.publish(msg)
            


            rospy.sleep(self.param.dt)
            self.step += 1

            if self.step > self.param.nbDataPoints:
                rospy.signal_shutdown("Mission complete")


        


if __name__ == "__main__":
    ergodic_node = ErgodicNode()
    rospy.spin()    