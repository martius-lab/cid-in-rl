import types

import gym
import numpy as np
import pyglet
from gym import spaces
from gym.utils import seeding

EPS = 1e-5


class OneDSlideEnv(gym.GoalEnv):
    """Goal-based 1D-env where an agent has to slide an object to a goal

    The agent has to slide a box to a goal location. There is a wall in the
    middle of the environment which the agent can not cross. Therefore, the
    final push of the agent has to be controlled such that the object exactly
    slides to the goal location.

    Observation:
        Type: Box(4)
        Num Observation                 Min         Max
        0   Agent Position             -Inf         Inf
        1   Agent Velocity             -Inf         Inf
        2   Box Position               -Inf         Inf
        3   Box Velocity               -Inf         Inf

    Actions:
        Type: Box(1)
        Num Action                      Min         Max
        0   Acceleration to apply      -1           1

    Reward:
        `0` if the center of the object is within goal distance, `-1`
        otherwise.

    Starting State:
        Agent is initialized randomly on the left half. The object is
        randomly initialized to the right of the agent, but within reach of the
        agent. The goal is initialized randomly on the right half.

    Episode Termination:
        No natural termination. Episode is supposed to be terminated after N
        steps.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, goal_width=1, friction=0.15, dt=0.2,
                 action_mult=1,
                 central_wall_stops_agent=True,
                 stop_agent_while_object_moves=False,
                 done_on_success=False,
                 trace_control=True):
        self.world_size = 10
        self.agent_size = 0.2
        self.agent_mass = 1
        self.obj_size = 0.2
        self.obj_mass = 0.5
        self.action_mult = action_mult
        self.friction = friction
        self.dt = dt

        self.central_wall_stops_agent = central_wall_stops_agent
        self.stop_agent_while_object_moves = stop_agent_while_object_moves
        self.done_on_success = done_on_success
        self.trace_control = trace_control

        self.max_speed = 5.0
        self.min_pos = -self.world_size / 2
        self.max_pos = self.world_size / 2
        self.central_wall = -0.5

        self.agent_min_initial_offset = -3.9  # Offset agent-center
        self.agent_range = 1.0  # Range around offset for random agent init
        self.obj_min_initial_offset = 1.5  # Offset object-agent
        self.obj_range = 0.5  # Range around offset for random object init
        self.goal_min_initial_offset = 2.5  # Offset goal-center
        self.goal_range = 2.0  # Range around goal for random goal init

        self.distance_threshold = goal_width / 2

        self.seed()

        # Set by `_set_initial_state`
        self.agent_pos = None
        self.agent_vel = None
        self.object_pos = None
        self.object_vel = None
        # Set by `_set_goal`
        self.goal_pos = None
        self.goal = None
        # Set by `step`
        self._last_info = None

        self._set_initial_state()
        self._set_goal(self._sample_goal())

        obs = self._get_obs()

        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=obs['achieved_goal'].shape,
                                    dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=obs['achieved_goal'].shape,
                                     dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape,
                                   dtype='float32'),
        ))

        self.viewer = None
        self.indicator_scale = None

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def _sample_goal(self):
        goal = np.zeros((1,), dtype=np.float32)
        goal += self.goal_min_initial_offset
        goal += self.rng.uniform(-self.goal_range, self.goal_range)

        return goal

    def _set_goal(self, goal):
        self.goal = goal
        self.goal_pos = goal

    def _set_initial_state(self):
        self.agent_pos = np.zeros((1,), dtype=np.float32)
        self.agent_pos += self.agent_min_initial_offset
        self.agent_pos += self.rng.uniform(-self.agent_range, self.agent_range)
        self.agent_vel = np.zeros((1,), dtype=np.float32)

        obj_pos = self.agent_pos.copy()
        obj_pos += self.obj_min_initial_offset
        obj_pos += self.rng.uniform(-self.obj_range, self.obj_range)
        self.obj_pos = obj_pos
        self.obj_vel = np.zeros((1,), dtype=np.float32)

    def _get_obs(self):
        return {
            'observation': np.concatenate((self.agent_pos,
                                           self.agent_vel,
                                           self.obj_pos,
                                           self.obj_vel)).astype(np.float32),
            'achieved_goal': np.array(self.obj_pos, dtype=np.float32),
            'desired_goal': np.array(self.goal, dtype=np.float32)
        }

    def reset(self):
        super(OneDSlideEnv, self).reset()

        self._set_initial_state()
        self._set_goal(self._sample_goal())

        return self._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = abs(achieved_goal - desired_goal)
        return -(dist > self.distance_threshold).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        dist = abs(achieved_goal - desired_goal)
        return (dist <= self.distance_threshold).astype(np.float32)

    def _wall_collision(self, pos, vel, wall, from_direction):
        # `from_direction == -1`: object coming from left
        # `from_direction ==  1`: object coming from right
        if -from_direction * (pos - wall) > 0:
            pos = np.zeros((1,), dtype=np.float32) + wall
            if vel != 0:
                vel = np.zeros((1,), dtype=np.float32)

        return pos, vel

    def _simulate_step(self, action):
        agent_pos = self.agent_pos.copy()
        agent_vel = self.agent_vel.copy()
        obj_pos = self.obj_pos.copy()
        obj_vel = self.obj_vel.copy()

        if not (self.stop_agent_while_object_moves and obj_vel == 0):
            dv = (action * self.action_mult / self.agent_mass) * self.dt
            agent_vel += dv
            agent_vel = np.clip(agent_vel, -self.max_speed, self.max_speed)

        agent_pos += agent_vel * self.dt

        # Agent-Object contact
        agent_obj_dist_before = self.agent_pos - self.obj_pos
        agent_obj_dist = agent_pos - obj_pos
        agent_object_contact = False
        if (np.sign(-agent_obj_dist_before) * agent_obj_dist
                + (self.agent_size + self.obj_size) / 2) >= 0:
            agent_object_contact = True
            obj_vel += self.agent_mass * agent_vel / self.obj_mass
            agent_pos = (obj_pos + np.sign(agent_obj_dist_before)
                         * (self.agent_size + self.obj_size) / 2)
            agent_vel = np.zeros((1,))

        if obj_vel > 0:
            obj_force = self.obj_mass * obj_vel / self.dt
            friction = (-np.sign(obj_force) * self.friction
                        * self.obj_mass * 9.81)
            obj_vel = max((obj_force + friction) * self.dt / self.obj_mass,
                          np.zeros((1,)))

        obj_pos += obj_vel * self.dt

        obj_pos, obj_vel = self._wall_collision(obj_pos, obj_vel, self.max_pos,
                                                from_direction=-1)
        if self.central_wall_stops_agent:
            agent_pos, agent_vel = self._wall_collision(agent_pos, agent_vel,
                                                        self.central_wall,
                                                        from_direction=-1)

        agent_pos, agent_vel = self._wall_collision(agent_pos, agent_vel,
                                                    self.max_pos,
                                                    from_direction=-1)
        agent_pos, agent_vel = self._wall_collision(agent_pos, agent_vel,
                                                    self.min_pos,
                                                    from_direction=1)

        return agent_pos, agent_vel, obj_pos, obj_vel, agent_object_contact

    def step(self, action):
        info = {}

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.trace_control:
            info['has_control'] = self.check_control()

        outputs = self._simulate_step(action)
        agent_pos, agent_vel, obj_pos, obj_vel, contact = outputs

        self.agent_pos = agent_pos
        self.agent_vel = agent_vel
        self.obj_pos = obj_pos
        self.obj_vel = obj_vel

        obs = self._get_obs()

        is_success = self._is_success(obs['achieved_goal'], self.goal)
        done = is_success if self.done_on_success else False

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        info['is_success'] = is_success
        info['is_contact'] = contact
        self._last_info = info

        return obs, reward, done, info

    def check_control(self):
        """Check if agent currently has control over object

        Control in this case means being able to perform an action that
        changes the state of the object through a contact. We test just
        the two extremal actions here, but this is not sufficient in
        general.
        """
        has_control = False
        for a in (self.action_space.high, self.action_space.low):
            _, _, obj_pos, obj_vel, contact = self._simulate_step(a)
            if contact and abs(obj_vel - self.obj_vel) > EPS:
                has_control = True
                break

        return has_control

    def set_indicator(self, scale: float):
        """Set the scale of the indicator

        The indicator is a purely visual element that is rendered above the
        agent to indicate the current value of something.
        """
        assert scale is None or scale >= 0
        self.indicator_scale = scale

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        line_y = 100
        scale = screen_width / self.world_size

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            def make_rect(height, width, color=None):
                l = (-width / 2) * scale
                r = (width / 2) * scale
                b = 0
                t = height * scale

                rect = rendering.FilledPolygon([(l, b), (l, t),
                                                (r, t), (r, b)])
                transform = rendering.Transform()
                rect.add_attr(transform)

                if color is not None:
                    rect._color.vec4 = (color[0], color[1], color[2], color[3])

                return rect, transform

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.render = types.MethodType(_custom_render, self.viewer)

            agent, self.agent_trans = make_rect(self.agent_size,
                                                self.agent_size)
            self.viewer.add_geom(agent)

            obj, self.obj_trans = make_rect(self.obj_size, self.obj_size,
                                            color=(0.0, 0.7, 0.3, 1.0))
            self.viewer.add_geom(obj)

            goal, self.goal_trans = make_rect(self.obj_size,
                                              self.distance_threshold * 2,
                                              color=(1.0, 0.0, 0.0, 0.5))
            self.viewer.add_geom(goal)

            indic, self.indic_trans = make_rect(self.agent_size,
                                                self.agent_size / 3,
                                                color=(0.2, 0.678, 1.0, 1.0))
            self.viewer.add_geom(indic)

            self.indic_line = rendering.Line((-101, 0), (-100, 0))
            self.indic_line.set_color(0, 0, 0)
            self.viewer.add_geom(self.indic_line)

            line = rendering.Line((0, line_y), (screen_width, line_y))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

            wall_pos = self.central_wall * scale + screen_width / 2
            line = rendering.Line((wall_pos, line_y), (wall_pos, line_y + 25))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

            self.labels = []
            self.contact_label = _write_text('',
                                             int(0.8 * screen_width),
                                             int(0.95 * screen_height))

            if self.trace_control:
                self.control_label = _write_text('',
                                                 int(0.8 * screen_width),
                                                 int(0.9 * screen_height))
            else:
                self.control_label = _write_text('Control: ?',
                                                 int(0.8 * screen_width),
                                                 int(0.9 * screen_height))
            self.labels = [self.contact_label, self.control_label]

        agent_x = self.agent_pos * scale + screen_width / 2
        self.agent_trans.set_translation(agent_x, line_y)

        obj_x = self.obj_pos * scale + screen_width / 2
        self.obj_trans.set_translation(obj_x, line_y)

        goal_x = self.goal_pos * scale + screen_width / 2
        self.goal_trans.set_translation(goal_x, line_y)

        if self.indicator_scale is not None:
            indic_y = line_y + 1.75 * scale * self.agent_size
            self.indic_trans.set_translation(agent_x, indic_y)
            self.indic_trans.set_scale(1, self.indicator_scale)
            self.indic_line.start = (agent_x - 0.5 * scale * self.agent_size,
                                     indic_y)
            self.indic_line.end = (agent_x + 0.5 * scale * self.agent_size,
                                   indic_y)
        else:
            self.indic_trans.set_translation(agent_x, -100)
            self.indic_line.start = (-101, 0)
            self.indic_line.end = (-100, 0)

        if self._last_info is None:
            contact = False
        else:
            contact = self._last_info['is_contact']
        self.contact_label.text = 'Contact: {}'.format(contact)

        if self.trace_control:
            if self._last_info is None:
                control = self.check_control()
            else:
                control = self._last_info['has_control']
            self.control_label.text = 'Control: {}'.format(control)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'),
                                  labels=self.labels)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def _write_text(text, pos_x, pos_y, size=12):
    return pyglet.text.Label(text,
                             font_name='Arial',
                             font_size=size,
                             color=(0, 0, 0, 255),
                             x=pos_x,
                             y=pos_y)


def _custom_render(self, return_rgb_array=False, labels=None):
    """Custom render method which supports drawing text labels"""
    pyglet.gl.glClearColor(1, 1, 1, 1)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    self.transform.enable()
    for geom in self.geoms:
        geom.render()
    for geom in self.onetime_geoms:
        geom.render()
    self.transform.disable()
    if labels is not None:
        for label in labels:
            label.draw()
    arr = None
    if return_rgb_array:
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1, :, 0:3]
    self.window.flip()
    self.onetime_geoms = []
    return arr if return_rgb_array else self.isopen


if __name__ == '__main__':
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random', action='store_true',
                        help='Random action')
    parser.add_argument('--dt', default=0.2, type=float,
                        help='Simulation time per step')
    parser.add_argument('--fps', type=int,
                        help='Frames per second')
    parser.add_argument('-s', '--steps', default=40, type=int,
                        help='Steps per episode')
    parser.add_argument('-e', '--episodes', default=10, type=int,
                        help='Number of episodes')

    args = parser.parse_args(sys.argv[1:])

    if args.fps is None:
        args.fps = 1 / args.dt

    env = OneDSlideEnv(dt=args.dt, action_mult=5, trace_control=True)
    env.seed(0)

    for _ in range(args.episodes):
        env.reset()
        env.render()

        for _ in range(args.steps):
            if args.random:
                a = env.action_space.sample()
            else:
                a = np.ones((1,), dtype=np.float32)
            env.step(a)
            time.sleep(1 / args.fps)
            env.render()

    env.close()
