#!/bin/bash

if [[ $# == 0 ]]; then
    echo "Usage: ./make_conda_env.sh <minimal|minimal-gpu|dev|dev-gpu> <env-name>"
    exit 1
fi

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

if [[ $1 == "minimal" ]]; then
    conda create --yes -n $2 python=3.7 pip scipy scikit-learn tqdm pytest
    conda activate $2
    conda install --yes pytorch=1.8.1 cpuonly -c pytorch
    pip install gym==0.18.0 mujoco-py gin-config==0.3.0 
elif [[ $1 == "minimal-gpu" ]]; then
    conda create --yes -n $2 python=3.7 pip scipy scikit-learn tqdm pytest
    conda activate $2
    conda install --yes pytorch=1.8.1 cudatoolkit=10.2 -c pytorch
    pip install gym==0.18.0 mujoco-py gin-config==0.3.0
elif [[ $1 == "dev" ]]; then
    conda create --yes -n $2 python=3.7 pip jupyter notebook matplotlib seaborn scipy scikit-learn pandas tqdm pytest
    conda activate $2
    conda install --yes pytorch=1.8.1 cpuonly -c pytorch
    pip install gym==0.18.0 mujoco-py gin-config==0.3.0 tensorboard 
elif [[ $1 == "dev-gpu" ]]; then
    conda create --yes -n $2 python=3.7 pip jupyter notebook matplotlib seaborn scipy scikit-learn pandas tqdm pytest
    conda activate $2
    conda install --yes pytorch=1.8.1 cudatoolkit=10.2 -c pytorch
    pip install gym==0.18.0 mujoco-py gin-config==0.3.0 tensorboard 
else
    echo "Usage: ./make_conda_env.sh <minimal|minimal-gpu|dev|dev-gpu> <env-name>"
    exit 1
fi
