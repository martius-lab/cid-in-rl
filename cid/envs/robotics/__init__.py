import gin

from cid.envs.robotics.contact_detection_wrapper \
    import ContactDetectionWrapper
from cid.envs.robotics.fetch_control_detection \
    import ControlDetectionWrapper
from cid.envs.robotics.fetch_info_wrapper import FetchInfoWrapper


@gin.configurable(blacklist=['env'])
def make_control_and_contact_fetch_env(env):
    env = ContactDetectionWrapper(env)
    env = ControlDetectionWrapper(env)

    return env


__all__ = [make_control_and_contact_fetch_env,
           ContactDetectionWrapper,
           ControlDetectionWrapper,
           FetchInfoWrapper]
