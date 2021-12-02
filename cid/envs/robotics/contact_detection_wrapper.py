"""Wrapper extracting contacts from Fetch* environments"""
import gin
import gym
import mujoco_py
import numpy as np

from cid.envs.robotics.fetch_contact_detection import \
    FetchPickAndPlaceContactDetectionEnv
from cid.envs.robotics.pick_and_place_rot_table import \
    FetchPickAndPlaceRotTableEnv

VALID_ENVS = (FetchPickAndPlaceContactDetectionEnv,
              FetchPickAndPlaceRotTableEnv)

MODEL_KEY_RIGHT_FINGER = 'robot0:r_gripper_finger_link'
MODEL_KEY_LEFT_FINGER = 'robot0:l_gripper_finger_link'
MODEL_KEY_OBJECT = 'object0'

NCONMAX = 100  # Maximum number of contacts stored in mjData.contact array

CONTACT_OFS = {
    'geom1': 0,
    'geom2': 1,
    # Contact force
    'cf1': 2,
    'cf2': 3,
    'cf3': 4,
    'cf4': 5,
    'cf5': 6,
    'cf6': 7,
}

CONTACT_SIZE = max(CONTACT_OFS.values()) + 1
SUBSTEP_SIZE = 1 + NCONMAX * CONTACT_SIZE

C_DEFINES = '''
#include <mjmodel.h>
#include <mujoco.h>

#define NCONMAX {nconmax}
#define CONTACT_SIZE {contact_size}
#define SUBSTEP_SIZE {substep_size}
#define OFS_GEOM1 {geom1}
#define OFS_GEOM2 {geom2}
#define OFS_CF1 {cf1}
#define OFS_CF2 {cf2}
#define OFS_CF3 {cf3}
#define OFS_CF4 {cf4}
#define OFS_CF5 {cf5}
#define OFS_CF6 {cf6}
'''.format(nconmax=NCONMAX,
           contact_size=CONTACT_SIZE,
           substep_size=SUBSTEP_SIZE,
           **CONTACT_OFS)

C_STORE_CONTACTS_FN = '''
void fun(const mjModel* m, mjData* d) {
    int substep = d->userdata[0];
    if (substep == %(n_substeps)s) {
        // Reset substep count if we are through one full step
        substep = 0;
        d->userdata[0] = 0;
    }

    int step_ofs = 1 + substep * SUBSTEP_SIZE;
    d->userdata[step_ofs] = d->ncon;

    for (int i = 0; i < d->ncon; i++) {
        int contact_ofs = 1 + i * CONTACT_SIZE;
        d->userdata[step_ofs + contact_ofs + OFS_GEOM1] = d->contact[i].geom1;
        d->userdata[step_ofs + contact_ofs + OFS_GEOM2] = d->contact[i].geom2;

        mjtNum contact_force[6];
        mj_contactForce(m, d, i, contact_force);
        d->userdata[step_ofs + contact_ofs + OFS_CF1] = contact_force[0];
        d->userdata[step_ofs + contact_ofs + OFS_CF2] = contact_force[1];
        d->userdata[step_ofs + contact_ofs + OFS_CF3] = contact_force[2];
        d->userdata[step_ofs + contact_ofs + OFS_CF4] = contact_force[3];
        d->userdata[step_ofs + contact_ofs + OFS_CF5] = contact_force[4];
        d->userdata[step_ofs + contact_ofs + OFS_CF6] = contact_force[5];
    }

    // Increase substep count
    d->userdata[0] += 1;
}
'''

C_CALLBACK = C_DEFINES + C_STORE_CONTACTS_FN


@gin.configurable(blacklist=['env'])
class ContactDetectionWrapper(gym.Wrapper):
    """Environment wrapper for Fetch* envs with an object

    Includes a flag in the info dict indicating whether robot gripper had
    contact with the object in the transition.
    """
    def __init__(self, env, force_threshold=30, return_diagnostics=False):
        super().__init__(env)
        self._force_threshold = force_threshold
        self._return_diagnostics = return_diagnostics

        env = self.unwrapped
        assert isinstance(env, VALID_ENVS), \
            ('`ContactDetectionWrapper` can only be applied to Fetch* envs '
             'which have been prepared for it.')

        self._sim = env.sim
        assert len(self._sim.data.contact) == NCONMAX
        self._contacts = ContactArrayWrapper(self._sim.data.userdata)

        callback = C_CALLBACK % {'n_substeps': self._sim.nsubsteps}
        self._sim.set_substep_callback(callback)

        id_r_finger = env.sim.model.geom_name2id(MODEL_KEY_RIGHT_FINGER)
        id_l_finger = env.sim.model.geom_name2id(MODEL_KEY_LEFT_FINGER)
        id_object = env.sim.model.geom_name2id(MODEL_KEY_OBJECT)

        robot_parts = (id_r_finger, id_l_finger)
        self._touch_pairs = ([(id_, id_object) for id_ in robot_parts]
                             + [(id_object, id_) for id_ in robot_parts])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self._return_diagnostics:
            diagnostics = []

        forces = {pair: 0 for pair in self._touch_pairs}
        for substep in range(1, self._sim.nsubsteps + 1):
            last_step = substep == self._sim.nsubsteps
            if last_step:
                n_contacts = self._sim.data.ncon
                contacts = [(c.geom1, c.geom2)
                            for c in self._sim.data.contact[:n_contacts]]
            else:
                contacts = self._contacts.get_contacts(substep)

            for idx, contact in enumerate(contacts):
                if contact in self._touch_pairs:
                    if last_step:
                        force = np.zeros((6,))
                        mujoco_py.functions.mj_contactForce(self._sim.model,
                                                            self._sim.data,
                                                            idx, force)
                    else:
                        force = self._contacts.get_contact_force(substep, idx)

                    force_magnitude = np.sqrt(force[0]**2
                                              + force[1]**2
                                              + force[2]**2)
                    forces[contact] += force_magnitude

                    if self._return_diagnostics:
                        diagnostics.append({
                            'substep': substep - 1,
                            'contact_idx': idx,
                            'contact_pair': contact,
                            'force': np.array(force),
                            'force_magnitude': force_magnitude
                        })

        is_contact = False
        for force in forces.values():
            if force >= self._force_threshold:
                is_contact = True
                break

        info['is_contact'] = is_contact

        if self._return_diagnostics:
            info['contact_diagnostics'] = diagnostics

        return observation, reward, done, info


class ContactArrayWrapper:
    """Convenience class extracting contact pairs from userdata"""
    def __init__(self, userdata):
        self._data = userdata

    def get_ncons(self, substep):
        return int(self._data[1 + substep * SUBSTEP_SIZE])

    def get_contact(self, substep, contact_idx):
        step_ofs = 1 + substep * SUBSTEP_SIZE
        contact_ofs = 1 + CONTACT_SIZE * contact_idx
        geom1 = self._data[step_ofs + contact_ofs + CONTACT_OFS['geom1']]
        geom2 = self._data[step_ofs + contact_ofs + CONTACT_OFS['geom2']]
        return int(geom1), int(geom2)

    def get_contacts(self, substep):
        ncons = self.get_ncons(substep)
        step_ofs = 1 + substep * SUBSTEP_SIZE

        contacts = []
        for idx in range(ncons):
            contact_ofs = step_ofs + 1 + CONTACT_SIZE * idx
            geom1 = self._data[contact_ofs + CONTACT_OFS['geom1']]
            geom2 = self._data[contact_ofs + CONTACT_OFS['geom2']]
            contacts.append((int(geom1), int(geom2)))

        return contacts

    def get_contact_force(self, substep, contact_idx):
        step_ofs = 1 + substep * SUBSTEP_SIZE
        contact_ofs = step_ofs + 1 + CONTACT_SIZE * contact_idx

        return (self._data[contact_ofs + CONTACT_OFS['cf1']],
                self._data[contact_ofs + CONTACT_OFS['cf2']],
                self._data[contact_ofs + CONTACT_OFS['cf3']],
                self._data[contact_ofs + CONTACT_OFS['cf4']],
                self._data[contact_ofs + CONTACT_OFS['cf5']],
                self._data[contact_ofs + CONTACT_OFS['cf6']])

    def get_contact_forces(self, substep):
        ncons = self.get_ncons(substep)

        return [self.get_contact_force(substep, idx)
                for idx in range(ncons)]
