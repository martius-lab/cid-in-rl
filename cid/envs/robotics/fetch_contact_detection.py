"""Verbatim copy of the Fetch environments under

https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/

The only difference is that we inject our own XML model files, which defines
the `nuserdata` variable we need for detecting the contacts.
"""
import os

import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env


class FetchPickAndPlaceContactDetectionEnv(fetch_env.FetchEnv, utils.EzPickle):
    # Need to use absolute path, otherwise `RobotEnv` attempts to use its own file
    MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'assets/fetch/pick_and_place_contact_detection.xml')

    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, self.MODEL_XML_PATH, has_object=True, block_gripper=False, 
            n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True, 
            target_offset=0.0, obj_range=0.15, target_range=0.15, 
            distance_threshold=0.05, initial_qpos=initial_qpos, 
            reward_type=reward_type)
        utils.EzPickle.__init__(self)
