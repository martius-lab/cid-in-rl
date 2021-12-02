import enum
import multiprocessing as mp
from typing import Dict

import numpy as np
import torch

from cid.rollouts.base import BaseRolloutGenerator


class ParallelRolloutGenerator(BaseRolloutGenerator):
    """Generator producing multiple environment rollouts in parallel

    Parallel rollouts are generated in worker processes, where each process has
    its own copy of the environment.
    """
    def __init__(self, n_parallel_rollouts, env_fn, agent_fn,
                 rollout_generator_fn, seed, gin_conf=None):
        assert n_parallel_rollouts > 0
        self._processes = []
        self._pipes = []

        ctx = mp.get_context('spawn')
        for idx in range(n_parallel_rollouts):
            worker_seed = seed + idx
            process, pipe = self._start_worker_process(ctx,
                                                       idx,
                                                       env_fn,
                                                       agent_fn,
                                                       rollout_generator_fn,
                                                       worker_seed,
                                                       gin_conf)
            self._processes.append(process)
            self._pipes.append(pipe)

    @staticmethod
    def _start_worker_process(ctx, worker_id, env_fn, agent_fn,
                              rollout_generator_fn, worker_seed,
                              gin_conf=None):
        parent_pipe, worker_pipe = ctx.Pipe(duplex=True)
        args = (worker_id,
                worker_pipe,
                parent_pipe,
                CloudpickleWrapper(env_fn),
                CloudpickleWrapper(agent_fn),
                CloudpickleWrapper(rollout_generator_fn),
                worker_seed,
                gin_conf)
        name = f'ParallelRolloutGenerator-Worker-{worker_id}'
        process = ctx.Process(target=worker_main,
                              name=name,
                              args=args)

        # If the main process crashes, we should not cause things to hang
        process.daemon = True

        process.start()
        worker_pipe.close()

        return process, parent_pipe

    def rollout(self, agent, n_steps, evaluate=False, render=False):
        for pipe in self._pipes:
            pipe.send((Commands.SYNC_PARAMS, agent.get_state()))

        for pipe in self._pipes:
            pipe.recv()

        for idx, pipe in enumerate(self._pipes):
            # Render only in first worker process
            render = render if idx == 0 else False
            pipe.send((Commands.ROLLOUT, (n_steps, evaluate, render)))

        # Receive tuple of (episode, stats) from each worker
        results = [pipe.recv() for pipe in self._pipes]

        episodes = {key: np.concatenate([res[0][key] for res in results])
                    for key in results[0][0]}
        stats = {key: [res[1][key] for res in results]
                 for key in results[0][1]}

        return episodes, stats

    def reset(self):
        for i, pipe in enumerate(self._pipes):
            pipe.send((Commands.RESET, None))

        obs = [pipe.recv() for pipe in self._pipes]

        return np.array(obs)

    def close(self):
        for pipe in self._pipes:
            pipe.send((Commands.CLOSE, None))

        for pipe in self._pipes:
            pipe.recv()
            pipe.close()

        for process in self._processes:
            process.join()

    @property
    def example_transition(self) -> Dict[str, np.ndarray]:
        self._pipes[0].send((Commands.EXAMPLE_TRANSITION, None))
        return self._pipes[0].recv()

    @property
    def transition_help(self) -> Dict[str, str]:
        self._pipes[0].send((Commands.TRANSITION_HELP, None))
        return self._pipes[0].recv()

    @property
    def observation_space(self) -> 'gym.Space':
        self._pipes[0].send((Commands.OBSERVATION_SPACE, None))
        return self._pipes[0].recv()

    @property
    def action_space(self) -> 'gym.Space':
        self._pipes[0].send((Commands.ACTION_SPACE, None))
        return self._pipes[0].recv()


def worker_main(worker_id, pipe, parent_pipe, env_fn, agent_fn,
                rollout_generator_fn, seed, gin_conf):
    """Entry-point of worker processes"""
    parent_pipe.close()

    if gin_conf is not None:
        # At the moment, gin configs are not propagated to new processes
        # automatically, so we have to reload the config here. Somewhat
        # inconvenient.
        import gin
        gin.parse_config(gin_conf)

    # Each process needs to be seeded individually
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = env_fn()
    env.seed(seed)

    rollout_gen = rollout_generator_fn(env)
    agent = agent_fn(rollout_gen.observation_space, rollout_gen.action_space)

    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == Commands.SYNC_PARAMS:
                agent.set_state(data)
                pipe.send(None)
            elif cmd == Commands.ROLLOUT:
                pipe.send(rollout_gen.rollout(agent, *data))
            elif cmd == Commands.RESET:
                pipe.send(rollout_gen.reset())
            elif cmd == Commands.EXAMPLE_TRANSITION:
                pipe.send(rollout_gen.example_transition)
            elif cmd == Commands.TRANSITION_HELP:
                pipe.send(rollout_gen.transition_help)
            elif cmd == Commands.OBSERVATION_SPACE:
                pipe.send(rollout_gen.observation_space)
            elif cmd == Commands.ACTION_SPACE:
                pipe.send(rollout_gen.action_space)
            elif cmd == Commands.CLOSE:
                pipe.send(None)
                break
            else:
                raise NotImplementedError(f'Unknown command {cmd}')
    except KeyboardInterrupt:
        print('ParallelRolloutWorker: got KeyboardInterrupt')
    finally:
        rollout_gen.close()


class Commands(enum.Enum):
    """Commands for communication between parent and worker processes"""
    ROLLOUT = 1
    SYNC_PARAMS = 2
    RESET = 3
    CLOSE = 4
    EXAMPLE_TRANSITION = 5
    TRANSITION_HELP = 6
    OBSERVATION_SPACE = 7
    ACTION_SPACE = 8


class CloudpickleWrapper:
    """Wrapper to cloudpickle functions

    Adapted from Gym
    """
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(ob)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
