import itertools
import pathlib

import gin
import gym
import numpy as np
import scipy.spatial
import tqdm

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

SENSOR_INFO_REACH = {
    'grip_pos': [0, 1, 2],
    'gripper_state': [3, 4],
    'grip_velp': [5, 6, 7],
    'gripper_vel': [8, 9]
}


@gin.configurable(blacklist=['env'])
class ControlDetectionWrapper(gym.Wrapper):
    """Environment wrapper for Fetch* envs with an object

    Includes a flag in the info dict indicating whether robot gripper can
    approximately control the object
    """
    def __init__(self, env, bounds_file=None, object_radius=0.025,
                 gripper_finger_one_step_y_range=0.0407,
                 gripper_finger_max_pos=0.054,
                 gripper_finger_extent=(0.0135, 0.007, 0.0385),
                 gripper_z_ofs=0.02):
        super().__init__(env)

        env = self.unwrapped
        assert isinstance(env, gym.envs.robotics.FetchEnv), \
            '`ControlDetectionWrapper` can only be applied to Fetch* envs'
        assert env.has_object, \
            '`ControlDetectionWrapper` can only be applied to object envs'

        self._object_radius = object_radius
        self._gripper_finger_one_step_y_range = gripper_finger_one_step_y_range
        self._gripper_finger_max_pos = gripper_finger_max_pos
        self._gripper_finger_extent = np.array(gripper_finger_extent)
        self._gripper_z_ofs = gripper_z_ofs

        if bounds_file is None:
            path = pathlib.Path(__file__).parent.parent.parent
            bounds_file = str(path / 'data/fetch_pnp_bounds.npz')

        with open(bounds_file, 'rb') as f:
            res = np.load(f)
            self._start_positions = res['start_positions']
            abs_distances = np.abs(res['end_positions']
                                   - self._start_positions[:, None])
            self._mov_ellipsoids = np.max(abs_distances, axis=1)

        self._kd_tree = scipy.spatial.KDTree(self._start_positions)
        self._last_obs = None

    def reset(self):
        obs = self.env.reset()
        self._last_obs = obs['observation']

        return obs

    @staticmethod
    def _check_in_ellipsoid(ellipsoid_pos, target_point, ellipsoid):
        point = target_point - ellipsoid_pos
        ellipsoid_eq = np.sum(point**2 / ellipsoid**2)

        return ellipsoid_eq <= 1

    def step(self, action):
        robot_pos = self._last_obs[SENSOR_INFO_PNP['grip_pos']]
        gripper_state = self._last_obs[SENSOR_INFO_PNP['gripper_vel']]
        gripper_vel = self._last_obs[SENSOR_INFO_PNP['gripper_state']]
        object_pos = self._last_obs[SENSOR_INFO_PNP['object_pos']]

        gripper_y_range = (np.max(gripper_state + gripper_vel)
                           + self._gripper_finger_one_step_y_range)
        gripper_y_range = np.clip(gripper_y_range, 0.0,
                                  self._gripper_finger_max_pos)
        ellipsoid_ext = (self._gripper_finger_extent +
                         self._object_radius
                         + np.array([0, gripper_y_range, self._gripper_z_ofs]))

        _, idx = self._kd_tree.query(robot_pos)
        ellipsoid_mov = self._mov_ellipsoids[idx]
        control = self._check_in_ellipsoid(robot_pos,
                                           object_pos,
                                           ellipsoid_mov + ellipsoid_ext)

        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs['observation']

        if 'is_contact' in info:
            control = control or info['is_contact']

        info['has_control'] = control

        return obs, reward, done, info


def move_gripper_to_pos(sim, target_pos, n_steps=15):
    gripper_target = target_pos
    gripper_rotation = np.array([1., 0., 1., 0.])
    sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(n_steps):
        sim.step()

    return sim.data.get_site_xpos('robot0:grip')


def perform_test(env, target_points, test_actions, n_targets=None):
    sim = env.sim
    start_positions = []
    end_positions = []

    for target in tqdm.tqdm(targets, 'Performing range test',
                            total=n_targets):
        env.reset()
        move_gripper_to_pos(sim, target)

        start_positions.append(sim.data.get_site_xpos('robot0:grip').copy())
        end_positions.append([])

        for action in test_actions:
            action = np.concatenate((action, [0]))
            obs, _, _, _ = env.step(action)

            end_pos = sim.data.get_site_xpos('robot0:grip').copy()
            end_positions[-1].append(end_pos)

            env.reset()
            move_gripper_to_pos(sim, target)

    return (np.array(start_positions, dtype=np.float32),
            np.array(end_positions, dtype=np.float32))


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./fetch_bounds.npz',
                        help='Path to output artifact')

    args = parser.parse_args(sys.argv[1:])

    env = gym.make('FetchReach-v1')
    sim = env.unwrapped.sim
    env.reset()

    initial_pos = sim.data.get_site_xpos('robot0:grip').copy()

    grid_sparse = []  # Sparser sampling for outer regions
    for low, high in zip(np.array([0.5, 0, 0.05]), np.array([1.55, 1.4, 1])):
        grid_sparse.append(np.linspace(low, high, 20))
    grid_dense = []  # Denser sampling for inner regions
    for low, high in zip(np.array([1.1, 0.5, 0.4]), np.array([1.5, 1.0, 0.9])):
        grid_dense.append(np.linspace(low, high, 50))

    actions = np.array([[-1, 0, 0], [1, 0, 0],
                        [0, -1, 0], [0, 1, 0],
                        [0, 0, -1], [0, 0, 1]], dtype=np.float32)
    targets = itertools.chain([initial_pos],
                              itertools.product(*grid_sparse),
                              itertools.product(*grid_dense))

    start_positions, end_positions = perform_test(env.unwrapped,
                                                  targets, actions,
                                                  n_targets=1 + 50**3 + 20**3)

    with open(args.path, 'wb') as f:
        np.savez(f,
                 start_positions=start_positions,
                 end_positions=end_positions)
