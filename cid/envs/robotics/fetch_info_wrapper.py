import gin
import gym
import numpy as np

SENSOR_INFO_PNP = {
    'grip_pos': [0, 1, 2],
    'object_pos': [3, 4, 5],
    'object_rel_pos': [6, 7, 8],
    'gripper_state': [9, 10],
    'object_rot': [11, 12, 13],
    'object_velp': [14, 15, 16],
    'object_velr': [17, 18, 19],
    'grip_velp': [20, 21, 22],
    'gripper_vel': [23, 24]
}


@gin.configurable(blacklist=['env'])
class FetchInfoWrapper(gym.Wrapper):
    """Wrapper adding more information to info dict"""
    def __init__(self, env, movement_eps=1e-4, above_table_z=0.43,
                 below_table_z=0.42):
        super().__init__(env)
        env = self.unwrapped
        assert isinstance(env, gym.envs.robotics.FetchEnv), \
            '`FetchInfoWrapper` can only be applied to Fetch* envs'

        self._movement_eps = movement_eps
        self._above_table_z = above_table_z
        self._below_table_z = below_table_z

        self._last_obs = None

    def reset(self):
        obs = self.env.reset()
        self._last_obs = obs['observation']

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        last_obj_pos = self._last_obs[SENSOR_INFO_PNP['object_pos']]
        obj_pos = obs['observation'][SENSOR_INFO_PNP['object_pos']]

        has_contact = 'is_contact' in info

        if np.linalg.norm(obj_pos - last_obj_pos) > self._movement_eps:
            info['obj_movement'] = True
            if has_contact:
                info['obj_movement_by_agent'] = info['is_contact']
        else:
            info['obj_movement'] = False
            if has_contact:
                info['obj_movement_by_agent'] = False

        x, y, z = obj_pos
        on_table = ((1.19 < x) & (x < 1.5) & (0.59 < y) & (y < 0.9))
        if ((0.03 < z < self._below_table_z and not on_table)
                or z > self._above_table_z):
            info['obj_air'] = True
            info['obj_air_above_table'] = z > self._above_table_z
            if has_contact:
                info['obj_air_by_agent'] = info['is_contact']
        else:
            info['obj_air'] = False
            info['obj_air_above_table'] = False
            if has_contact:
                info['obj_air_by_agent'] = False

        self._last_obs = obs['observation']

        return obs, reward, done, info
