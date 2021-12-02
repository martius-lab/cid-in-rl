"""FetchPickAndPlace with Rotating Table

FetchPickAndPlace and its assets are part of OpenAI Gym, licensed under MIT.
"""
import os

import gym
import numpy as np
import scipy.integrate
from gym import utils
from gym.envs.robotics import fetch_env, rotations

_ASSETS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'assets/fetch')
_MODEL_XML_PATH = os.path.join(_ASSETS_PATH, 'pick_and_place_rot_table.xml')
# Version of XML file where robot is able to interact with table
_FLEX_MODEL_XML_PATH = os.path.join(_ASSETS_PATH,
                                    'pick_and_place_rot_table_flex.xml')


class FetchPickAndPlaceRotTableEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 steps_rot=25,
                 steps_still=5,
                 forward_angle=45,
                 target_also_in_the_air=True,
                 target_only_in_the_air=False,
                 robot_table_interaction=False,
                 p_air_goal=0.5):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.41, 1., 0., 0., 0.],
        }

        self.target_only_in_the_air = target_only_in_the_air
        self.p_air_goal = p_air_goal
        self.steps_rot = steps_rot
        self.steps_still = steps_still
        self.forward_angle = np.radians(forward_angle)

        if robot_table_interaction:
            xml_path = _FLEX_MODEL_XML_PATH
        else:
            xml_path = _MODEL_XML_PATH

        fetch_env.FetchEnv.__init__(self, xml_path,
                                    has_object=True,
                                    block_gripper=False,
                                    n_substeps=20,
                                    gripper_extra_height=0.2,
                                    target_in_the_air=target_also_in_the_air,
                                    target_offset=0.0,
                                    obj_range=0.15,
                                    target_range=0.15,
                                    distance_threshold=0.05,
                                    initial_qpos=initial_qpos,
                                    reward_type=reward_type)
        gym.utils.EzPickle.__init__(self)

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.initial_sim_time = self.sim.data.time
        self.period_T = self.steps_rot * self.dt
        self.const_T = self.steps_still * self.dt
        self.moving_T = self.period_T - self.const_T
        self.integral = self._compute_integral()
        self._last_table_rot_delta = None

    def _get_obs(self):
        sim_time = self.sim.data.time - self.initial_sim_time
        self._last_table_rot_delta = self._get_table_rot_delta(sim_time)

        table_xmat = self.sim.data.get_body_xmat('table1')
        table_rot = rotations.mat2euler(table_xmat)[-1]
        table_velr = self.sim.data.get_site_xvelr('table1')[-1] * self.dt

        table_state = [np.sin(table_rot), np.cos(table_rot),
                       table_velr, self._last_table_rot_delta]

        obs = super()._get_obs()
        obs['observation'] = np.concatenate((obs['observation'], table_state))

        return obs

    def _get_table_rot_delta(self, sim_time):
        t = sim_time % self.period_T

        if t <= self.moving_T:
            v = np.abs(np.sin(t * np.pi / self.moving_T))
        else:
            v = 0

        return v / self.integral * self.dt * self.forward_angle

    @property
    def table_rot(self):
        mat_table = self.sim.data.get_body_xmat('table1')
        return np.degrees(rotations.mat2euler(mat_table)[-1])

    def _compute_integral(self):
        def integrand(t):
            return np.abs(np.sin(np.pi * t / self.moving_T))

        return scipy.integrate.quad(integrand, 0, self.moving_T)[0]

    def _set_action(self, action):
        assert action.shape == (4,)

        # Ensure that we don't change the action outside of this scope
        action = action.copy()

        pos_ctrl = action[:3]
        gripper_ctrl = action[3]

        # Limit maximum change in position
        pos_ctrl *= 0.05
        # Fixed rotation of the end effector, expressed as a quaternion
        rot_ctrl = [1., 0., 1., 0.]

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl,
                                [self._last_table_rot_delta]])

        # Apply action to simulation.
        gym.envs.robotics.utils.ctrl_set_action(self.sim, action)
        gym.envs.robotics.utils.mocap_set_action(self.sim, action)

    def _sample_goal(self):
        gripper_ofs = self.np_random.uniform(-self.target_range,
                                             self.target_range,
                                             size=3)
        goal = self.initial_gripper_xpos[:3] + gripper_ofs
        goal += self.target_offset
        goal[2] = self.height_offset

        if (self.target_in_the_air and
                self.np_random.uniform() < self.p_air_goal):
            r = self.np_random.uniform(0, 0.45)
            goal[2] += r
        elif self.target_only_in_the_air:
            goal[2] += self.np_random.uniform(0.1, 0.45)

        return goal.copy()


def _hide_mocaps(sim):
    for body_idx1, val in enumerate(sim.model.body_mocapid):
        if val != -1:
            for geom_idx, body_idx2 in enumerate(sim.model.geom_bodyid):
                if body_idx1 == body_idx2:
                        sim.model.geom_rgba[geom_idx, 3] = 0


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--flex', action='store_true', default=False,
                        help='Robot can interact with table')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='Save RGB images instead of rendering to window')
    parser.add_argument('-s', '--steps', default=100, type=int,
                        help='Steps per episode')
    parser.add_argument('-e', '--episodes', default=10, type=int,
                        help='Number of episodes')
    args = parser.parse_args(sys.argv[1:])

    if args.save_images:
        from PIL import Image

    env = FetchPickAndPlaceRotTableEnv(robot_table_interaction=args.flex)
    env.seed(args.seed)

    if args.save_images:
        _hide_mocaps(env.unwrapped.sim)

    for ep in range(args.episodes):
        env.reset()
        env.render()

        for step in range(args.steps):
            a = env.action_space.sample()
            a = np.zeros_like(a)
            obs, _, _, _ = env.step(a)

            if args.save_images:
                img = env.render(mode='rgb_array', width=2000, height=2000)
                img = Image.fromarray(img)
                img.save(f'img_{ep}_{step}.png')
            else:
                env.render()

    env.close()
