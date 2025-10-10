import numpy as np
from numpy.linalg import norm
import abc
import logging
import crowd_sim.envs.utils.collisions as Collisions
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section, obstacles, bounds):
        """
        Base class for robot and human. Keeps original kinematics behavior.
        Adds bounds/obstacle projection only.
        """
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = self.policy.kinematics if self.policy is not None else None

        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

        # references to map data
        self.obstacles = obstacles
        self.bounds = bounds

        # bump flags (optional, for reward shaping/logging)
        self._last_hit_wall = False
        self._last_hit_obst = False

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """Sample agent radius and v_pref attribute from certain distribution."""
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        next_px, next_py = self.compute_position(action, self.time_step)
        if self.kinematics == 'holonomic':
            next_vx, next_vy = action.vx, action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius,
                         self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px, self.py = position[0], position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def set_goal_position(self, position):
        self.gx, self.gy = position[0], position[1]

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx, self.vy = velocity[0], velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """Compute action from observation via policy."""
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        """
        Compute intended next pose (as before) and then project against bounds/obstacles.
        No side effects, no changes to velocity/heading here.
        """
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            nx = self.px + action.vx * delta_t
            ny = self.py + action.vy * delta_t
        else:
            theta_new = self.theta + action.r
            nx = self.px + np.cos(theta_new) * action.v * delta_t
            ny = self.py + np.sin(theta_new) * action.v * delta_t

        fx, fy, _, _ = Collisions.move_and_project(
            self.px, self.py, nx, ny, self.radius, self.bounds, self.obstacles
        )
        return fx, fy

    def step(self, action):
        """
        Perform an action and update the state.
        Keeps original kinematics updates. Only records bumps.
        """
        self.check_validity(action)

        # recompute intended next, capture bump flags via projection
        if self.kinematics == 'holonomic':
            nx = self.px + action.vx * self.time_step
            ny = self.py + action.vy * self.time_step
        else:
            theta_new = self.theta + action.r
            nx = self.px + np.cos(theta_new) * action.v * self.time_step
            ny = self.py + np.sin(theta_new) * action.v * self.time_step

        fx, fy, hit_wall, hit_obst = Collisions.move_and_project(
            self.px, self.py, nx, ny, self.radius, self.bounds, self.obstacles
        )

        # commit pose
        self.px, self.py = fx, fy

        # record bumps (for reward shaping/logging if you want)
        self._last_hit_wall = bool(hit_wall)
        self._last_hit_obst = bool(hit_obst)

        # ORIGINAL velocity/heading updates (unchanged)
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def bumped_wall(self):
        return self._last_hit_wall

    def bumped_obstacle(self):
        return self._last_hit_obst

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius
