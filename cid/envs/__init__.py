import gin
import gym


@gin.configurable()
def make_gym_env(name, wrapper_env_fn=None):
    env = gym.make(name)
    if wrapper_env_fn is not None:
        env = wrapper_env_fn(env)

    return env


@gin.configurable()
def nesting_wrapper_fn(env, outter_wrapper_fn, inner_wrapper_fn):
    wrapped_env = inner_wrapper_fn(env)
    return outter_wrapper_fn(wrapped_env)


@gin.configurable()
def get_env_reward_fn(env_fn):
    env = env_fn()

    assert hasattr(env, 'compute_reward')

    def reward_fn(achieved_goals, goals):
        return env.compute_reward(achieved_goals, goals, None)

    env.close()

    return reward_fn


gym.envs.registration.register(
    id='OneDSlide-v0',
    entry_point='cid.envs.one_d_slide:OneDSlideEnv',
    max_episode_steps=50,
)

gym.envs.registration.register(
    id='OneDSlide-v1',
    entry_point='cid.envs.one_d_slide:OneDSlideEnv',
    max_episode_steps=30,
    kwargs=dict(dt=0.1, action_mult=5)
)

gym.envs.registration.register(
    id='FetchPickAndPlaceContactDetection-v1',
    entry_point=('cid.envs.robotics.fetch_contact_detection'
                 ':FetchPickAndPlaceContactDetectionEnv'),
    max_episode_steps=50,
)

gym.envs.registration.register(
    id='FetchPickAndPlaceRotTable-v0',
    entry_point=('cid.envs.robotics.pick_and_place_rot_table'
                 ':FetchPickAndPlaceRotTableEnv'),
    max_episode_steps=50,
)

gym.envs.registration.register(
    id='FetchPickAndPlaceRotTable-Harder-v0',
    entry_point=('cid.envs.robotics.pick_and_place_rot_table'
                 ':FetchPickAndPlaceRotTableEnv'),
    max_episode_steps=50,
    kwargs=dict(p_air_goal=0.9)
)
