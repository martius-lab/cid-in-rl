# Causal Influence Detection for Improving Efficiency in Reinforcement Learning

This repository contains the code release for the paper ["Causal Influence Detection for Improving Efficiency in Reinforcement Learning"](https://arxiv.org/abs/2106.03443), published at NeurIPS 2021. 

This work was done by Maximilian Seitzer, Bernhard Schölkopf and Georg Martius at the [Autonomous Learning Group](https://al.is.tuebingen.mpg.de/), Max-Planck Institute for Intelligent Systems.

If you make use of our work, please use the citation information [below](https://github.com/martius-lab/cid-in-rl#citation).

## Abstract

Many reinforcement learning (RL) environments consist of independent entities that interact sparsely. In such environments, RL agents have only limited influence over other entities in any particular situation. Our idea in this work is that learning can be efficiently guided by knowing when and what the agent can influence with its actions. To achieve this, we introduce a measure of situation-dependent causal influence based on conditional mutual information and show that it can reliably detect states of influence. We then propose several ways to integrate this measure into RL algorithms to improve exploration and off-policy learning. All modified algorithms show strong increases in data efficiency on robotic manipulation tasks. 

## Setup

Use `make_conda_env.sh` to create a Conda environment with minimal dependencies:

```
./make_conda_env.sh minimal cid_in_rl
```

or recreate the environment used to get the results (more dependencies than necessary):

```
conda env create -f orig_environment.yml
```

Activate the environment with `conda activate cid_in_rl`.

## Experiments

### Causal Influence Detection

To reproduce the causal influence detection experiment, you will need to [download the used datasets here](https://edmond.mpdl.mpg.de/imeji/exportServlet?format=file&id=http://edmond.mpdl.mpg.de/imeji/item/zdZNHpttyCakKTuK).
Extract them into the folder `data/`. 
The most simple way to run all experiments is to use the included Makefile (this will take a long time):

```
make -C experiments/1-influence
```

The results will be in the folder `./data/experiments/1-influence/`.

You can also train a single model, for example

```
python -m cid.influence_estimation.train_model \
        --log-dir logs/eval_fetchpickandplace 
        --no-logging-subdir --seed 0 \
        --memory-path data/fetchpickandplace/memory_5k_her_agent_v2.npy \
        --val-memory-path data/fetchpickandplace/val_memory_2kof5k_her_agent_v2.npy \
        experiments/1-influence/pickandplace_model_gaussian.gin
```

which will train a model on FetchPickPlace, and put the results in `logs/eval_fetchpickandplace`.

To evaluate the CAI score performance of the model on the validation set, use 

```
python experiments/1-influence/pickandplace_cmi.py 
    --output-path logs/eval_fetchpickandplace 
    --model-path logs/eval_fetchpickandplace
    --settings-path logs/eval_fetchpickandplace/eval_settings.gin \
    --memory-path data/fetchpickandplace/val_memory_2kof5k_her_agent_v2.npy 
    --variants var_prod_approx
```

### Reinforcement Learning

The RL experiments can be reproduced using the settings in `experiments/2-prioritization`, `experiments/3-exploration`, `experiments/4-other`.

To do so, run 

```
python -m cid.train <path-to-settings-gin-file>
```

By default, the output will be in the folder `./logs`.


## Codebase Overview

- `cid/algorithms/ddpg_agent.py` contains the DDPG agent
- `cid/envs` contains new environments
  * `cid/envs/one_d_slide.py` implements the 1D-Slide dataset
  * `cid/envs/robotics/pick_and_place_rot_table.py` implements the RotatingTable environment
  * `cid/envs/robotics/fetch_control_detection.py` contains the code for deriving ground truth control labels for FetchPickAndPlace
- `cid/influence_estimation` contains code for model training, evaluation and computing the causal influence score
  * `cid/influence_estimation/train_model.py` is the main model training script
  * `cid/influence_estimation/eval_influence.py` evaluates a trained model for its classification performance
  * `cid/influence_estimation/transition_scorers` contains code for computing the CAI score
- `cid/memory/` contains the replay buffers, which handle prioritization and exploration bonuses
  * `cid/memory/mbp` implements CAI (ours)
  * `cid/memory/her` implements Hindsight Experience Replay
  * `cid/memory/ebp` implements Energy-Based Hindsight Experience Prioritization
  * `cid/memory/per` implements Prioritized Experience Replay
- `cid/models` contains Pytorch model implementations
  * `cid/bnn.py` contains the implementation of VIME
- `cid/play.py` lets a trained RL agent run in an environment
- `cid/train.py` is the main RL training script

## Citation

Please use the following citation if you make use of our work:

```
@inproceedings{Seitzer2021CID,
  title = {Causal Influence Detection for Improving Efficiency in Reinforcement Learning},
  author = {Seitzer, Maximilian and Sch{\"o}lkopf, Bernhard and Martius, Georg},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2021)},
  month = dec,
  year = {2021},
  url = {https://arxiv.org/abs/2106.03443},
  month_numeric = {12}
}
```

## License

This implementation is licensed under the MIT license.

The robotics environments were adapted from [OpenAI Gym](https://github.com/openai/gym/) under MIT license.
The VIME implementation was adapted from https://github.com/alec-tschantz/vime under MIT license.