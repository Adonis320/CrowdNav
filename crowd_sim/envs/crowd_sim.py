import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import crowd_sim.envs.utils.collisions as Collisions
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        self.bounds = (-6.0, 6.0, -6.0, 6.0)   # <-- default bounds
        self.obstacles = []                    # <-- default empty list

    def add_rectangle_obstacle(self, xmin, xmax, ymin, ymax):
        self.obstacles.append((xmin, xmax, ymin, ymax))

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if config.has_option('env', 'xmin'):
            self.bounds = (config.getfloat('env', 'xmin'),
                           config.getfloat('env', 'xmax'),
                           config.getfloat('env', 'ymin'),
                           config.getfloat('env', 'ymax'))
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.obstacles = []
        if self.config.has_option('env', 'obstacles'):
            import yaml
            raw = self.config.get('env', 'obstacles')
            try:
                obs_list = yaml.safe_load(raw) or []
                for o in obs_list:
                    xmin = float(o['xmin']); xmax = float(o['xmax'])
                    ymin = float(o['ymin']); ymax = float(o['ymax'])
                    # if you have add_rectangle_obstacle, use it; else append tuple:
                    # self.add_rectangle_obstacle(xmin, xmax, ymin, ymax)
                    self.obstacles.append((xmin, xmax, ymin, ymax))
            except Exception as e:
                logging.warning(f"Failed to parse env.obstacles: {e}")
        logging.info(f"Loaded {len(self.obstacles)} obstacles")

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side
        """
        # --- local helper: is (x,y) legal w.r.t. bounds+obstacles with margin=radius ---
        def _pose_ok(x, y, r):
            xmin, xmax, ymin, ymax = self.bounds
            if not (xmin + r <= x <= xmax - r and ymin + r <= y <= ymax - r):
                return False
            # if projecting out of obstacles would move the point, it was inside => invalid
            _, _, bumped = Collisions.project_outside_obstacles(x, y, r, self.obstacles)
            return not bumped

        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for _ in range(human_num):
                self.humans.append(self.generate_square_crossing_human())  # generate_* already checks legality
        elif rule == 'circle_crossing':
            self.humans = []
            for _ in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())  # generate_* already checks legality
        elif rule == 'mixed':
            # mix different training simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []

            if static:
                # randomly initialize static objects; ensure inside bounds and outside obstacles
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans', self.obstacles, self.bounds)
                    # pick a legal dummy point well outside the scene so it never draws
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)

                for _ in range(human_num):
                    human = Human(self.config, 'humans', self.obstacles, self.bounds)
                    sign = -1 if np.random.random() > 0.5 else 1
                    # sample until legal and not colliding with existing agents
                    tries = 0
                    while True:
                        tries += 1
                        # original sampling box (kept), but we reject if illegal
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height

                        if not _pose_ok(px, py, human.radius):
                            if tries > 200:
                                # fallback: nudge into legal region deterministically
                                px, py, _ = Collisions.project_outside_obstacles(px, py, human.radius, self.obstacles)
                                px, py = Collisions.project_within_bounds(px, py, human.radius, self.bounds)
                                break
                            continue

                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break

                    # static => goal == start
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)

            else:
                # the first two humans: circle crossing; the rest: square crossing
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")


    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans', self.obstacles, self.bounds)
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise

            # NEW: reject if outside bounds or inside obstacles
            if not self._outside_all_obstacles(px, py, human.radius):
                continue

            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        gx, gy = -px, -py
        if not self._outside_all_obstacles(gx, gy, human.radius):
            gx, gy, _ = Collisions.project_outside_obstacles(gx, gy, human.radius, self.obstacles)
            gx, gy = Collisions.project_within_bounds(gx, gy, human.radius, self.bounds)
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans', self.obstacles, self.bounds)
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            # NEW: reject if illegal
            if not self._outside_all_obstacles(px, py, human.radius):
                continue

            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            # NEW: reject if illegal
            if not self._outside_all_obstacles(gx, gy, human.radius):
                continue

            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, **kwargs):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        options = kwargs.get("options", {})
        if "phase" in options:
            phase = options["phase"]
        if "test_case" in options:
            test_case = options["test_case"]
        else:
            test_case = None
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            # initial desired positions
            # initial desired positions
            sx, sy = 0.0, -self.circle_radius
            gx, gy = 0.0,  self.circle_radius

            # FIX: bounds first, then obstacles, with clearance + fallback
            sx, sy = self._legalize_point(sx, sy, self.robot.radius)
            gx, gy = self._legalize_point(gx, gy, self.robot.radius)

            self.robot.set(sx, sy, gx, gy, 0, 0, np.pi / 2)

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans', self.obstacles, self.bounds) for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step
            px, py = Collisions.project_within_bounds(agent.px, agent.py, agent.radius, self.bounds)
            gx, gy = Collisions.project_within_bounds(agent.gx, agent.gy, agent.radius, self.bounds)
            px, py, _ = Collisions.project_outside_obstacles(px, py, agent.radius, self.obstacles)
            gx, gy, _ = Collisions.project_outside_obstacles(gx, gy, agent.radius, self.obstacles)
            agent.set_position((px, py))
            agent.set_goal_position((gx, gy))

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # --- robot-vs-human collision detection (uses actual next positions) ---
        dmin = float('inf')
        collision = False

        # robot's projected next position (respects bounds/obstacles)
        r_next_x, r_next_y = self.robot.compute_position(action, self.time_step)
        r_dx = r_next_x - self.robot.px
        r_dy = r_next_y - self.robot.py

        for human, h_act in zip(self.humans, human_actions):
            # human's projected next position
            h_next_x, h_next_y = human.compute_position(h_act, self.time_step)
            h_dx = h_next_x - human.px
            h_dy = h_next_y - human.py

            # relative motion over this step: human relative to robot
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            ex = px + (h_dx - r_dx)
            ey = py + (h_dy - r_dy)

            # closest distance between the two agent circles over the step (capsule clearance)
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
        # ----------------------------------------------------------------------

        # collision detection between humans (informational only)
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal (use robot's projected next position)
        end_position = np.array([r_next_x, r_next_y])
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        # robot vs obstacle clearance contribution
        if not collision:
            x0, y0 = self.robot.px, self.robot.py
            x1, y1 = float(end_position[0]), float(end_position[1])
            for rect in self.obstacles:
                d_rect = Collisions.capsule_to_rect_clearance(x0, y0, x1, y1, rect, self.robot.radius)
                if d_rect < 0:
                    collision = True
                    break
                dmin = min(dmin, d_rect)

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = False          # end episode on collision (optional; set False to continue)
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            dist_to_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position()))
            reward = -dist_to_goal * 0.01   # scale factor to keep numbers reasonable
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # --- only change here: do not advance agents on collision ---
            if not collision:
                # update all agents normally
                self.robot.step(action)
                for i, human_action in enumerate(human_actions):
                    self.humans[i].step(human_action)
            else:
                # freeze robot so it doesn't "drift" next step
                self.robot.vx = 0.0
                self.robot.vy = 0.0
                # (optional) also freeze humans
                # for h in self.humans:
                #     h.vx = 0.0
                #     h.vy = 0.0
            # ------------------------------------------------------------

            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info



    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def _draw_bounds_and_obstacles(ax):
            xmin, xmax, ymin, ymax = self.bounds
            from matplotlib.patches import Rectangle
            # bounds
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, linestyle='--', linewidth=1.5, alpha=0.8))
            # obstacles
            for (oxmin, oxmax, oymin, oymax) in self.obstacles:
                ax.add_patch(Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=True, alpha=0.15, linewidth=1.0))

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(self.bounds[0], self.bounds[1])
            ax.set_ylim(self.bounds[2], self.bounds[3])
            _draw_bounds_and_obstacles(ax) 
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(self.bounds[0], self.bounds[1])
            ax.set_ylim(self.bounds[2], self.bounds[3])
            _draw_bounds_and_obstacles(ax)    
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == "video":
            import matplotlib
            matplotlib.use("Agg")       # force non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib import animation
            import matplotlib.lines as mlines

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(self.bounds[0], self.bounds[1])
            ax.set_ylim(self.bounds[2], self.bounds[3])
            _draw_bounds_and_obstacles(ax)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color="red", marker="*", linestyle="None",
                                markersize=15, label="Goal")
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color="yellow")
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # humans
            human_positions = [[state[1][j].position for j in range(len(self.humans))]
                            for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                    for i in range(len(self.humans))]
            for h in humans:
                ax.add_artist(h)

            def update(frame_num):
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]

            anim = animation.FuncAnimation(fig, update, frames=len(self.states),
                                        interval=self.time_step * 1000, blit=False)

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            plt.close(fig)      # <- do not display

        else:
            raise NotImplementedError
    
    def _in_bounds_with_margin(self, x, y, r):
        xmin, xmax, ymin, ymax = self.bounds
        return (x >= xmin + r) and (x <= xmax - r) and (y >= ymin + r) and (y <= ymax - r)

    def _outside_all_obstacles(self, x, y, r):
        # reuse collisions helper by trying a single-step projection and checking if it moves
        x2, y2, bumped = Collisions.project_outside_obstacles(x, y, r, self.obstacles)
        return not bumped and self._in_bounds_with_margin(x, y, r)
    
    def _legalize_point(self, x, y, r, max_iters=12):
        """
        Clamp to bounds, then push out of any rectangle obstacles with a tiny clearance.
        If still illegal (e.g., wedged), jitter-search nearby valid spots.
        """
        eps = 1e-4  # tiny clearance so 'on the line' is treated as inside
        # 1) clamp to bounds
        x, y = Collisions.project_within_bounds(x, y, r, self.bounds)

        # 2) repeatedly push out of obstacles (inflated by r+eps)
        for _ in range(max_iters):
            x2, y2, bumped = Collisions.project_outside_obstacles(x, y, r + eps, self.obstacles)
            if not bumped:
                return x2, y2
            x, y = x2, y2  # continue in case we got pushed into another rect

        # 3) jitter-search around the current spot if still colliding
        step0 = max(0.5 * r, 0.05)
        for ring in range(1, 6):           # 5 rings
            step = step0 * ring
            for k in range(16):            # 16 angles per ring
                ang = 2.0 * np.pi * k / 16.0
                tx = x + step * np.cos(ang)
                ty = y + step * np.sin(ang)
                tx, ty = Collisions.project_within_bounds(tx, ty, r, self.bounds)
                _, _, bumped = Collisions.project_outside_obstacles(tx, ty, r + eps, self.obstacles)
                if not bumped:
                    return tx, ty

        # 4) give up gracefully: return the best we have
        return x, y
