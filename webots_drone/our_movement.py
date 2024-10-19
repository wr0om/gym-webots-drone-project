#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:19:44 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import numpy as np
import traceback
import os
import sys
import math

from webots_drone.utils import check_flight_area
from webots_drone.utils import compute_distance
from webots_drone.utils import min_max_norm
from webots_drone.utils import decode_image
from webots_drone.utils import emitter_send_json
from webots_drone.utils import receiver_get_json
from webots_drone.utils import compute_risk_distance

sys.path.append(os.environ['WEBOTS_HOME'] + "/lib/controller/python")
from controller import Supervisor




def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

# Webots environment controller
class WebotsSimulation(Supervisor):
    """
    Main class to control the Webots simulation scene.

    In order to work this class, a Robot node must be present in the Webots
    scenario with the supervisor option turned on. For this case the Robot
    node is considered as the RL-agent configured with the Emitter and
    Receiver nodes in order to get and send the states and actions,
    respectively, working as the Remote Control of the drone.
    Additionally, this class is responsible to randomize the fire size and
    location.
    Also, consider the implementation of a default keyboard control as human
    interface for testing purpose.
    """

    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 35.0 #30.0          # P constant of the pitch PID.

    MAX_YAW_DISTURBANCE = 2.4 #0.4
    MAX_PITCH_DISTURBANCE = -2.5 #-1
    # Precision between the target position and the robot position in meters
    target_precision = 0.5

    def __init__(self):
        super(WebotsSimulation, self).__init__()
        # simulation timestep
        self.timestep = int(self.getBasicTimeStep())
        self.image_shape = (240, 400, 4)
        self._data = dict()
        self.drone_name = 'Drone'

        self.enemy_drone_names = ['EnemyDrone1']
        self.enemy_drone_start_positions = {'EnemyDrone1': [5, 5, 15]}
        self.enemy_drone_target_positions = {'EnemyDrone1': [5, 5, 15]}

        self.enemy_drone_paths = {'EnemyDrone1': [[5, 5, 15], [5, 10, 15], [10, 10, 15], [7, 7, 15]]}
        self.enemy_drone_current_target = {'EnemyDrone1': 0} # next index in the path
        self.enemy_drone_began_paths = {'EnemyDrone1': False}
        self.enemy_drone_current_positions = {'EnemyDrone1': [0] * 6} # x, y, z, roll, pitch, yaw
        
        self.epsilon_radius = 0.5
        self.target_index = 0
        # save all the drones in some sort of list, and control them separately using the drone functions here, comunicate using names


        # actions value boundaries
        self.limits = self.get_control_ranges()
        # runtime vars
        self.init_nodes()
        self.init_comms()

    @property
    def is_running(self):
        """Get if the simulation is running."""
        return self.SIMULATION_MODE_PAUSE != self.simulationGetMode()

    def pause(self):
        """Pause the Webots's simulation."""
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

    def play(self):
        """Start the Webots's simulation in real time mode."""
        self.simulationSetMode(self.SIMULATION_MODE_REAL_TIME)

    def play_fast(self):
        """Start the Webots's simulation in fast mode."""
        self.simulationSetMode(self.SIMULATION_MODE_FAST)

    def seed(self, seed=None):
        """Set seed for the numpy.random and WorldInfo node, None default."""
        self.np_random = np.random.RandomState(seed)
        # world_node = self.getFromDef('World')
        # world_node.getField('randomSeed').setSFInt32(
        #     0 if seed is None else seed)
        return seed

    @staticmethod
    def get_control_ranges():
        """The control limits to manipulate the angles and altitude."""
        control_ranges = np.array([np.pi / 12.,     # roll
                                   np.pi / 12.,     # pitch
                                   np.pi ,           # yaw
                                   5.               # altitude
                                   ])
        return np.array([control_ranges * -1,  # low limits
                         control_ranges])      # high limits

    def get_flight_area(self, altitude_limits=[11, 75]):
        area_size = self.getFromDef('FlightArea').getField('size').getSFVec2f()
        area_size = [fs / 2 for fs in area_size]  # size from center
        flight_area = [[fs * -1 for fs in area_size], area_size]
        flight_area[0].append(altitude_limits[0])
        flight_area[1].append(altitude_limits[1])
        return flight_area

    def init_comms(self):
        """Initialize the communication nodes."""
        self.action = self.getDevice('ActionEmitter')  # channel 6
        self.state = self.getDevice('StateReceiver')  # channel 4
        self.state.enable(self.timestep)
        return self

    def init_areas(self):
        # Forest area
        forest_shape = self.getFromDef('ForestArea').getField('shape')
        self.forest_area = []

        for i in range(forest_shape.getCount()):
            self.forest_area.append(forest_shape.getMFVec2f(i))
        self.forest_area = np.asarray(self.forest_area)

        return self.forest_area

    def init_target_node(self):
        # Fire vars
        #target_node = self.getFromDef('Target')
        target_node = self.getFromDef('FireSmoke')
        self.target_node = dict(
            node=target_node,
            get_height=lambda: target_node.getField('fireHeight').getSFFloat(),
            get_radius=lambda: target_node.getField('fireRadius').getSFFloat(),
            get_pos=lambda: np.array(
                target_node.getField('translation').getSFVec3f()),
            set_height=target_node.getField('fireHeight').setSFFloat,
            set_radius=target_node.getField('fireRadius').setSFFloat,
            set_pos=target_node.getField('translation').setSFVec3f
        )
        self.risk_distance = compute_risk_distance(
            self.target_node['get_height'](), self.target_node['get_radius']())

    def init_drone_nodes(self):
        # Drone vars
        drone_node = self.getFromDef(self.drone_name)
        self.drone_node = dict(
            node=drone_node,
            get_pos=lambda: np.array(
                drone_node.getField('translation').getSFVec3f()),
            set_pos=drone_node.getField('translation').setSFVec3f
        )
        self.enemy_drone_nodes = []
        for enemy_drone in self.enemy_drone_names:
            enemy_drone_node = self.getFromDef(enemy_drone)
            enemy_drone_node = dict(
                node=enemy_drone_node,
                get_pos=lambda: np.array(
                    enemy_drone_node.getField('translation').getSFVec3f()),
                set_pos=enemy_drone_node.getField('translation').setSFVec3f
            )
            self.enemy_drone_nodes.append(enemy_drone_node)

    def init_nodes(self):
        """Initialize the target and drone nodes' information."""
        self.init_areas()
        self.init_target_node()
        self.init_drone_nodes()

    def reset(self):
        """Reset the Webots simulation.

        Set the simulation mode with the constant SIMULATION_MODE_PAUSE as
        defined in the Webots documentation.
        Reset the fire and drone nodes at the starting point, and restart
        the controllers simulation.
        """
        if self.is_running:
            self.state.disable()  # prevent to receive data
            #self.target_node['node'].restartController()
            self.drone_node['node'].restartController()
            self.simulationReset()
            self.simulationResetPhysics()
            # stop simulation
            self.one_step()  # step to process the reset
            self.pause()
            self.state.enable(self.timestep)
            self._data = dict()

    def one_step(self):
        """Do a Robot.step(timestep)."""
        self.step(self.timestep)

    def set_fire_dimension(self, fire_height=None, fire_radius=None):
        """
        Set the FireSmoke Node's height and radius.

        :param float fire_height: The fire's height, default is 2.
        :param float fire_radius: The fire's radius, default is 0.5

        :return float, float: the settled height and radius values.
        """
        if fire_height is None:
            fire_height = self.np_random.uniform(2., 13.)
        if fire_radius is None:
            fire_radius = self.np_random.uniform(0.5, 3.)

        # FireSmoke node fields
        self.target_node['set_height'](float(fire_height))
        self.target_node['set_radius'](float(fire_radius))

        # correct position in Z axis and update risk_distance value
        fire_pos = self.target_node['get_pos']()
        fire_pos[2] = fire_height * 0.5  # update height
        self.target_node['set_pos'](list(fire_pos))
        self.risk_distance = compute_risk_distance(fire_height, fire_radius)

        return (fire_height, fire_radius), self.risk_distance

    def set_fire_position(self, fire_pos=None):
        """
        Set the FireSmoke Node's position in the scenario.

        Set a desired node position value or generated a new random one inside
        the scenario's forest area if no input is given.

        :param list pos: The [X, Z] position values where locate the node, if
            no values are given a random one is generated instead.
            Default is None.
        """
        fire_radius = self.target_node['get_radius']()
        fire_p = self.target_node['get_pos']()  # current position
        # get forest limits
        X_range = [self.forest_area[:, 0].min(), self.forest_area[:, 0].max()]
        Y_range = [self.forest_area[:, 1].min(), self.forest_area[:, 1].max()]
        if fire_pos is None:
            # randomize position
            fire_p[0] = self.np_random.uniform(fire_radius - abs(X_range[0]),
                                               X_range[1] - fire_radius)
            fire_p[1] = self.np_random.uniform(fire_radius - abs(Y_range[0]),
                                               Y_range[1] - fire_radius)
        else:
            fire_p[0] = fire_pos[0]
            fire_p[1] = fire_pos[1]

        # ensure fire position inside the forest
        fire_p[0] = np.clip(fire_p[0], X_range[0], X_range[1])
        fire_p[1] = np.clip(fire_p[1], Y_range[0], Y_range[1])

        # set new position
        self.target_node['set_pos'](list(fire_p))

        return fire_p

    def set_fire(self, fire_pos=None, fire_height=None, fire_radius=None,
                 dist_threshold=0.):
        self.set_fire_dimension(fire_height, fire_radius)
        new_fire_pos = self.set_fire_position(fire_pos)

        # avoid to the fire appears near the drone's initial position
        must_do = 0
        directions = [(-1, 1), (1, 1),
                      (1, -1), (-1, -1)]
        while self.get_target_distance() <= self.get_risk_distance(dist_threshold):
            # randomize position offset
            offset = self.np_random.uniform(0.1, 1.)
            new_fire_pos[0] += offset * directions[must_do % 4][0]
            new_fire_pos[1] += offset * directions[must_do % 4][1]
            must_do += 1
            self.set_fire_position(new_fire_pos)

    def get_drone_pos(self):
        """Read the current drone position from the node's info."""
        return self.drone_node['get_pos']()

    def get_target_pos(self):
        """Read the current target position from the node's info."""
        return self.target_node['get_pos']()

    def get_target_distance(self):
        """Compute the drone's distance to the fire."""
        fire_position = self.get_target_pos()
        drone_position = self.get_drone_pos()
        # consider only xy coordinates
        fire_position[2] = drone_position[2]
        # Squared Euclidean distance
        distance = compute_distance(drone_position, fire_position)
        return distance

    def get_risk_distance(self, threshold=0.):
        return self.risk_distance + threshold

    def read_data(self):
        """Read the data sent by the drone's Emitter node.

        Capture and translate the drones sent data with the Receiver node.
        This data is interpreted as the drones states.

        Loop over all the drones names to capture all the data sent by them, to capture the states of all drones in the simulation.
        """
        all_drone_names = [self.drone_name] + self.enemy_drone_names
        received_drone_names = {name: False for name in all_drone_names}

        # capture all drones data each time
        while not all(received_drone_names.values()):
            # capture UAV sensors ()
            uav_state, emitter_info = receiver_get_json(self.state) # one packet

            if len(uav_state.keys()) == 0:
                return self._data

            drone_name = uav_state['drone_name']

            timestamp = uav_state['timestamp']
            orientation = uav_state['orientation']
            angular_velocity = uav_state['angular_velocity']
            position = uav_state['position']
            speed = uav_state['speed']
            north_rad = uav_state['north']
            dist_sensors = list()

            # Normalize distance sensor values
            for idx, sensor in uav_state['dist_sensors'].items():
                if sensor[2] == sensor[1] == sensor[0] == 0.:
                    continue
                s_val = min_max_norm(sensor[0],
                                    a=0, b=1,
                                    minx=sensor[1], maxx=sensor[2])
                dist_sensors.append(s_val)

            if type(uav_state['image']) == str and uav_state['image'] == "NoImage":
                img = np.zeros(self.image_shape)
            else:
                img = decode_image(uav_state['image'])

            self._data[drone_name] = dict(timestamp=timestamp,
                            orientation=orientation,
                            angular_velocity=angular_velocity,
                            position=position,
                            speed=speed,
                            north_rad=north_rad,
                            dist_sensors=dist_sensors,
                            motors_vel=uav_state['motors_vel'],
                            image=img,
                            emitter=emitter_info,
                            rc_position=self.getSelf().getPosition(),
                            target_position=self.target_node['get_pos'](),
                            target_dim=[self.target_node['get_height'](),
                                        self.target_node['get_radius']()])
            self.enemy_drone_current_positions[drone_name] = [self._data[drone_name]['position'][0], self._data[drone_name]['position'][1], self._data[drone_name]['position'][2],
                                                             self._data[drone_name]['orientation'][0], self._data[drone_name]['orientation'][1], self._data[drone_name]['orientation'][2]]
            received_drone_names[drone_name] = True

    def get_data(self, drone_name):
        return self._data.copy()[drone_name]

    def send_data(self, data, drone_name=None):
        # send data and do a Robot.step
        command = data
        command['timestamp'] = self.getTime()
        command['drone_name'] = drone_name #TODO: change to corresponding drone
        emitter_send_json(self.action, command)
        self.one_step()  # step to process the action
        self.read_data()

    def sync(self):
        # sync data
        while len(self._data.keys()) == 0:
            self.one_step()
            self.read_data()
        # sync orientation
        command = dict(disturbances=self.limits[1].tolist())
        all_drone_names = [self.drone_name] + self.enemy_drone_names
        for drone_name in all_drone_names:
            self.send_data(command, drone_name=drone_name)

    def __del__(self):
        """Stop simulation when is destroyed."""
        try:
            self.reset()
        except Exception as e:
            print('ERROR: unable to reset the environment!')
            traceback.print_tb(e.__traceback__)
            print(e)

    def clip_action(self, action, flight_area, drone_name):
        """Check drone position and orientation to keep inside FlightArea."""
        # clip action values
        action_clip = np.clip(action, self.limits[0], self.limits[1])
        roll_angle, pitch_angle, yaw_angle, altitude = action_clip

        # check area contraints
        info = self.get_data(drone_name)


        north_rad = info["north_rad"]
        out_area = check_flight_area(info["position"], flight_area)

        is_north = np.pi / 2. > north_rad > -np.pi / 2.  # north
        is_east = north_rad < 0
        orientation = [is_north,        # north
                       not is_north,    # south
                       is_east,         # east
                       not is_east]     # west
        movement = [pitch_angle > 0.,  # north - forward
                    pitch_angle < 0.,  # south - backward
                    roll_angle > 0.,   # east - right
                    roll_angle < 0.]   # west - left

        if out_area[0]:
            if ((orientation[0] and movement[0])
                    or (orientation[1] and movement[1])):  # N,S
                pitch_angle = 0.

            if ((orientation[2] and movement[3])
                    or (orientation[3] and movement[2])):  # E,W
                roll_angle = 0.

        if out_area[1]:
            if ((orientation[0] and movement[1])
                    or (orientation[1] and movement[0])):  # N,S
                pitch_angle = 0.

            if ((orientation[2] and movement[2])
                    or (orientation[3] and movement[3])):  # E,W
                roll_angle = 0.

        if out_area[2]:
            if ((orientation[0] and movement[2])
                    or (orientation[1] and movement[3])):  # N,S
                roll_angle = 0.

            if ((orientation[2] and movement[0])
                    or (orientation[3] and movement[1])):  # E,W
                pitch_angle = 0.

        if out_area[3]:
            if ((orientation[0] and movement[3])
                    or (orientation[1] and movement[2])):  # N,S
                roll_angle = 0.

            if ((orientation[2] and movement[1])
                    or (orientation[3] and movement[0])):  # E,W
                pitch_angle = 0.

        if ((out_area[4] and altitude > 0)  # ascense
                or (out_area[5] and altitude < 0)):  # descense
            altitude = 0.

        return roll_angle, pitch_angle, yaw_angle, altitude
    
    def init_enemy_drones_positions(self):
        reached_drone_names = {name: False for name in self.enemy_drone_names}

        while not all(reached_drone_names.values()):
            self.read_data()
            for enemy_drone in self.enemy_drone_names:
                drone_data = self.get_data(enemy_drone)
                drone_pos = drone_data['position']
                drone_orientation = drone_data['orientation']
                drone_roll, drone_pitch, drone_yaw = drone_orientation

                drone_angular_velocity = drone_data['angular_velocity']
                drone_speed = drone_data['speed']

                drone_start_pos = self.enemy_drone_start_positions[enemy_drone]

                x, y, z = drone_pos
                a, b, c = drone_start_pos

                # calculate disturbance needed to reach start position
                # Calculate position errors
                delta_x = a - x
                delta_y = b - y
                delta_z = c - z

                # check if drone is in epsilon radius sphere of start position
                if (delta_x**2 + delta_y**2 + delta_z**2) < self.epsilon_radius**2:
                    reached_drone_names[enemy_drone] = True # TODO: make sure they stand still when they reach the position

                # Calculate roll, pitch, yaw angles and altitude
                roll_angle = 0
                pitch_angle = np.arctan2(delta_z, np.sqrt(delta_x**2 + delta_y**2))
                yaw_angle = np.arctan2(delta_y, delta_x)
                
                # Calculate the deltas for roll, pitch, yaw, and altitude
                delta_roll = 0
                delta_pitch = pitch_angle - drone_pitch
                delta_yaw = yaw_angle - drone_yaw
                delta_altitude = delta_z

                # Populate the action array with the calculated disturbances
                action = [delta_roll, delta_pitch, delta_yaw, delta_altitude]
            

                action = controller.clip_action(action, controller.get_flight_area(), enemy_drone)
                print("Action: ", action)
                disturbances = dict(disturbances=action)
                # perform action
                controller.send_data(disturbances, drone_name=enemy_drone)



    def move_to_target(self, drone_name, verbose_movement=False, verbose_target=True):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """

        current_position = self.enemy_drone_current_positions[drone_name]
        target_position = self.enemy_drone_paths[drone_name][self.enemy_drone_current_target[drone_name]]

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(target_position, current_position[0:2])]):
            
            len_path = len(self.enemy_drone_paths[drone_name])
            if self.enemy_drone_current_target[drone_name] == len_path - 1:
                self.enemy_drone_current_target[drone_name] = len_path - 2
            elif self.enemy_drone_current_target[drone_name] == 0:
                self.enemy_drone_current_target[drone_name] = 1
            else:
                # randomly choose the next target +1 or -1
                self.enemy_drone_current_target[drone_name] += np.random.choice([-1, 1])

            target_position = self.enemy_drone_paths[drone_name][self.enemy_drone_current_target[drone_name]]
            self.enemy_drone_began_paths[drone_name] = True

            if verbose_target: print("New target: ", target_position)


        # This will be in [-pi;pi]
        tmp = np.arctan2(target_position[1] - current_position[1], target_position[0] - current_position[0])
        # This is now in [-2pi;2pi]
        angle_left = tmp - current_position[5]
        # Normalize turn angle to [-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi: angle_left -= 2 * np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # non proportional and decreasing function
        pitch_disturbance = clamp(np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((target_position[0] - current_position[0]) ** 2) + ((target_position[1] - current_position[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance


    def new_init_enemy_drones_positions(self):
        reached_drone_names = {name: False for name in self.enemy_drone_names}
        init_drone_positions = {name: True for name in self.enemy_drone_names}
        while not all(reached_drone_names.values()):
            self.read_data()
            roll_disturbance, yaw_disturbance, pitch_disturbance = 0, 0, 0
            for enemy_drone in self.enemy_drone_names:
                drone_data = self.get_data(enemy_drone)
                drone_pos = drone_data['position']
                drone_orientation = drone_data['orientation']
                roll, pitch, yaw = drone_orientation

                roll_acceleration, pitch_acceleration, _ = drone_data['angular_velocity']
                drone_speed = drone_data['speed']

                drone_start_pos = self.enemy_drone_start_positions[enemy_drone]

                x, y, z = drone_pos
                self.enemy_drone_current_positions[enemy_drone] = [x, y, z, roll, pitch, yaw]
                altitude = z
                a, b, c = drone_start_pos
                target_altitude = c

                # calculate disturbance needed to reach start position
                # Calculate position errors
                delta_x = a - x
                delta_y = b - y
                delta_z = c - z

                # check if drone is in epsilon radius sphere of start position
                if (delta_x**2 + delta_y**2 + delta_z**2) < self.epsilon_radius**2:
                    reached_drone_names[enemy_drone] = True # TODO: make sure they stand still when they reach the position

                if delta_z < 1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target([[a, b]], init_drone_positions[enemy_drone], enemy_drone)
                    init_drone_positions[enemy_drone] = False
            
                roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
                pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
                yaw_input = yaw_disturbance
                clamped_difference_altitude = clamp(target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
                vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

                front_left_motor_input =  vertical_input - yaw_input + pitch_input - roll_input
                front_right_motor_input = vertical_input + yaw_input + pitch_input + roll_input
                rear_left_motor_input =  vertical_input + yaw_input - pitch_input - roll_input
                rear_right_motor_input = vertical_input - yaw_input - pitch_input + roll_input

                action = [front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input]
                disturbances = dict(disturbances=action)
                # perform action
                controller.send_data(disturbances, drone_name=enemy_drone)


    def control_enemy_drones(self):
        # here continue the path using waypoints like in new_init_enemy_drones_positions with move_to_target
        self.read_data()
        for enemy_drone in self.enemy_drone_names:
            roll_disturbance, yaw_disturbance, pitch_disturbance = 0, 0, 0
            drone_data = self.get_data(enemy_drone)
            drone_pos = drone_data['position']
            drone_orientation = drone_data['orientation']
            roll, pitch, yaw = drone_orientation
            roll_acceleration, pitch_acceleration, yaw_acceleration = drone_data['angular_velocity']
            drone_speed = drone_data['speed']

            target_position = self.enemy_drone_paths[enemy_drone][self.enemy_drone_current_target[enemy_drone]]
            x, y, z = drone_pos
            self.enemy_drone_current_positions[enemy_drone] = [x, y, z, roll, pitch, yaw]
            altitude = z
            a, b, c = target_position
            target_altitude = c

            # calculate disturbance needed to reach target position
            # Calculate position errors
            delta_x = a - x
            delta_y = b - y
            delta_z = c - z

            # # check if drone is in epsilon radius sphere of target position
            # if (delta_x**2 + delta_y**2 + delta_z**2) < self.epsilon_radius**2:
            #     reached_drone_names[enemy_drone] = True # TODO: make sure they stand still when they reach the position

            print(f"target position: {target_position}, current position: {drone_pos}")
            if delta_z < 1:
                yaw_disturbance, pitch_disturbance = self.move_to_target(enemy_drone)
        
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input =  vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input =  vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = vertical_input - yaw_input - pitch_input + roll_input

            action = [front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input]
            disturbances = dict(disturbances=action)
            # perform action
            controller.send_data(disturbances, drone_name=enemy_drone)

if __name__ == '__main__':
    import cv2
    import datetime
    from webots_drone.reward import compute_vector_reward
    # from webots_drone.utils import compute_target_orientation
    # from webots_drone.envs.preprocessor import info2image
    # from webots_drone.reward import compute_visual_reward

    def print_control_keys():
        """Display manual control message."""
        print("You can control the drone with your computer keyboard:")
        print("IMPORTANT! The Webots 3D window must be selected to work!")
        print("- 'up': move forward.")
        print("- 'down': move backward.")
        print("- 'right': strafe right.")
        print("- 'left': strafe left.")
        print("- 'w': increase the target altitude.")
        print("- 's': decrease the target altitude.")
        print("- 'd': turn right.")
        print("- 'a': turn left.")
        print("- 'q': exit.")

    def get_kb_action(kb):
        # capture control data
        key = kb.getKey()

        run_flag = True
        take_shot = False
        roll_angle = 0.
        pitch_angle = 0.
        yaw_angle = 0.  # drone.yaw_orientation
        altitude = 0.  # drone.target_altitude

        while key > 0:
            # roll
            if key == kb.LEFT:
                roll_angle = controller.limits[0][0]
            elif key == kb.RIGHT:
                roll_angle = controller.limits[1][0]
            # pitch
            elif key == kb.UP:
                pitch_angle = controller.limits[1][1]
            elif key == kb.DOWN:
                pitch_angle = controller.limits[0][1]
            # yaw
            elif key == ord('D'):
                yaw_angle = controller.limits[0][2]
            elif key == ord('A'):
                yaw_angle = controller.limits[1][2]
            # altitude
            elif key == ord('W'):
                altitude = controller.limits[1][3]  # * 0.1
            elif key == ord('S'):
                altitude = controller.limits[0][3]  # * 0.1
            # quit
            elif key == ord('Q'):
                print('Terminated')
                run_flag = False
            # take photo
            elif key == ord('P'):
                print('Camera frame saved')
                take_shot = True
            key = kb.getKey()

        action = [roll_angle, pitch_angle, yaw_angle, altitude]
        action = controller.clip_action(action, controller.get_flight_area(), controller.drone_name)
        return action, run_flag, take_shot

    def run(controller, show=True):
        """Run controller's main loop.

        Capture the keyboard and translate into fixed float values to variate
        the 3 different angles and the altitude, optionally an image captured
        from the drone's camera can be presented in a new window.
        The pitch and roll angles are variated in +-pi/12.,
        the yaw angle in +-pi/360. and the altitude in +-5cm.
        The control keys are:
            - ArrowUp:      +pitch
            - ArrowDown:    -pitch
            - ArrowLeft:    -roll
            - ArrowRight:   +roll
            - W:            +altitude
            - S:            -altitude
            - A:            +yaw
            - D:            +yaw
            - Q:            EXIT

        :param bool show: Set if show or not the image from the drone's camera.
        """
        # keyboard interaction
        print_control_keys()
        kb = controller.getKeyboard()
        kb.enable(controller.timestep)

        # Start simulation with random FireSmoke position
        goal_threshold = 5.
        fire_pos = [50, -50]
        fire_dim = [7., 3.5]
        altitude_limits = [11., 75.]
        controller.seed()
        controller.set_fire(fire_pos=fire_pos, fire_height=fire_dim[0],
                            fire_radius=fire_dim[1],
                            dist_threshold=goal_threshold)
        controller.play()
        controller.sync()
        run_flag = True
        take_shot = False
        frame_skip = 25
        step = 0
        accum_reward = 0

        # get enemy drones in position
        #controller.init_enemy_drones_positions()
        # controller.new_init_enemy_drones_positions()

        # capture initial state
        state = controller.get_data(controller.drone_name)
        next_state = controller.get_data(controller.drone_name)

        print('Fire scene is running')
        while (run_flag):  # and drone.getTime() < 30):
            
            controller.control_enemy_drones()

            # check if all enemy drones have begun their paths
            if all([value for key, value in controller.enemy_drone_began_paths.items()]):
                if take_shot:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    cv2.imwrite(f'photos/picture_{timestamp}.png', state['image'])

                state = next_state.copy()
                # capture action
                action, run_flag, take_shot = get_kb_action(kb)
                disturbances = dict(disturbances=action)
                # perform action
                controller.send_data(disturbances, drone_name=controller.drone_name)
                # capture state
                next_state = controller.get_data(controller.drone_name)
            #     if show and next_state is not None:
            #         cv2.imshow("Drone's live view", next_state['image'])
            #         cv2.waitKey(1)
            #     step += 1
            # if show:
            #     cv2.destroyAllWindows()

    # run controller
    try:
        controller = WebotsSimulation()
        run(controller)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        controller.reset()
