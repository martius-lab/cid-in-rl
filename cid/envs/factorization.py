from collections import OrderedDict

import gin
import numpy as np


@gin.configurable
def one_d_slide(state):
    return OrderedDict(x=state[..., :2], o=state[..., 2:])


@gin.configurable
def one_d_slide_action(state_action):
    s = state_action
    return OrderedDict(x=s[..., :2], o=s[..., 2:4], a=s[..., 4:5])


@gin.configurable
def one_d_slide_state_action_next_object_pos(state_action_next_obj_pos):
    s = state_action_next_obj_pos
    return OrderedDict(x=s[..., :2], o=s[..., 2:4],
                       a=s[..., 4:5], o_next=s[..., 5:6])


@gin.configurable
def one_d_slide_state_next_object_pos(state_action_next_obj_pos):
    s = state_action_next_obj_pos
    return OrderedDict(x=s[..., :2], o=s[..., 2:4], o_next=s[..., 5:6])


@gin.configurable
def one_d_slide_object_pos(state):
    return OrderedDict(o=state[..., 2:3])


_FETCH_W_OBJ_AGENT_POS = [0, 1, 2]
_FETCH_W_OBJ_AGENT = [0, 1, 2, 20, 21, 22]
_FETCH_W_OBJ_AGENT_VEL = [20, 21, 22]
_FETCH_W_OBJ_OBJ = [3, 4, 5, 14, 15, 16, 11, 12, 13, 17, 18, 19]
_FETCH_W_OBJ_OBJ_POS = [3, 4, 5]
_FETCH_W_OBJ_OBJ_VELP = [14, 15, 16]
_FETCH_W_OBJ_GRIPPER = [9, 10, 23, 24]
_FETCH_W_OBJ_AGENT_OBJECT_POS = [0, 1, 2, 3, 4, 5]
_FETCH_W_OBJ_REL_POS = [6, 7, 8]


@gin.configurable
def fetch_with_object(state):
    obj = state[..., _FETCH_W_OBJ_OBJ]
    # Object velocity is relative to agent velocity
    obj[..., 3:6] += state[..., _FETCH_W_OBJ_AGENT_VEL]
    return OrderedDict(x=state[..., _FETCH_W_OBJ_AGENT],
                       o=obj,
                       xg=state[..., _FETCH_W_OBJ_GRIPPER])


@gin.configurable
def fetch_with_object_action(state):
    obj = state[..., _FETCH_W_OBJ_OBJ]
    # Object velocity is relative to agent velocity
    obj[..., 3:6] += state[..., _FETCH_W_OBJ_AGENT_VEL]
    return OrderedDict(x=state[..., _FETCH_W_OBJ_AGENT],
                       o=obj,
                       xg=state[..., _FETCH_W_OBJ_GRIPPER],
                       a=state[..., 25:29])


@gin.configurable
def fetch_with_object_object_pos(state):
    return OrderedDict(o=state[..., _FETCH_W_OBJ_OBJ_POS])


@gin.configurable
def flat_fetch_with_object_object_pos(state):
    return state[..., _FETCH_W_OBJ_OBJ_POS]


@gin.configurable
def flat_fetch_with_object_agent_object_pos(state):
    return state[..., _FETCH_W_OBJ_AGENT_OBJECT_POS]


@gin.configurable
def flat_fetch_with_object(state):
    return state[...,
                 _FETCH_W_OBJ_AGENT + _FETCH_W_OBJ_OBJ + _FETCH_W_OBJ_GRIPPER]


_FETCH_W_OBJ_STDS = np.array(
    [0.126301482, 0.155359909, 0.0978268459,    # grip pos
     0.105776899, 0.115924396, 0.0401872732,    # obj pos
     0.128538832, 0.163579434, 0.104058281,     # grip-obj
     0.0193517357, 0.0193344094,                # gripper
     0.617467821, 0.258399606, 0.496820241,     # obj rot
     0.004573488, 0.006080753, 0.004768900,     # obj velp (not grip-obj velp!)
     0.0713069811, 0.0532399304, 0.0547376908,  # obj velr
     0.0111851683, 0.0116869211, 0.0107023669,  # grip-velp
     0.0204342902, 0.0204684436]                # gripper vel
)


@gin.configurable
def fetch_with_object_state_noise_fn(state, noise, noise_level,
                                     only_object_noise=False):
    agent_vel = state[..., _FETCH_W_OBJ_AGENT_VEL]
    state = state.copy()

    if only_object_noise:
        selection = _FETCH_W_OBJ_OBJ
    else:
        selection = slice(None, None)

    n = noise[..., selection]
    std = _FETCH_W_OBJ_STDS[selection]
    state[..., selection] += noise_level * std * n

    # Recompute relative position
    state[..., _FETCH_W_OBJ_REL_POS] = (state[..., _FETCH_W_OBJ_OBJ_POS]
                                        - state[..., _FETCH_W_OBJ_AGENT_POS])
    # Recompute relative speed: first add old agent velocity to get noisy
    # object velocity, then subtract noisy agent velocity to get correct
    # relative speed between noisy measurements
    state[..., _FETCH_W_OBJ_OBJ_VELP] = (state[..., _FETCH_W_OBJ_OBJ_VELP]
                                         + agent_vel
                                         - state[..., _FETCH_W_OBJ_AGENT_VEL])

    return state


@gin.configurable
def fetch_with_object_ag_noise_fn(ag, noise, noise_level):
    n = noise[..., _FETCH_W_OBJ_OBJ_POS]
    std = _FETCH_W_OBJ_STDS[_FETCH_W_OBJ_OBJ_POS]
    return ag + noise_level * std * n
