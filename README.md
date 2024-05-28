# MInCo: Mitigating Information Conflicts in Visual Model-based Reinforcement Learning

## Introduction
This is a PyTorch implementation for the MInCo algorithm. MInCo is a visual model-based reinforcement learning method that mitigating information conflicts.


## Instructions

1. Install [Mujoco](https://www.roboti.us/index.html) 2.1.0

2. Create an environment 
   ```
   conda create -n minco python=3.8
   conda activate minco
   ```  

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```


## Distracted DMC experiments

Use one of the following commands to train an agent on distracted Walker Walk. To train on other distracted DMC environments,
replace `walker-run` with `{domain}-{task}`:

```
# MInCo
python experiments/train.py --algo minco --env_id dmc_distracted-walker-run --expr_name benchmark --seed 0 --a 8e-6 --prior_train_steps 5 --b 5 --c 0.015 --cross_inv_dynamics True

# RePo
python experiments/train.py --algo repo --env_id dmc_distracted-walker-run --expr_name benchmark --seed 0

# Dreamer
python experiments/train.py --algo dreamer --env_id dmc_distracted-walker-run --expr_name benchmark --seed 0
```

## Visualization
![walker run](demo/minco_walker_run.gif)

## Acknowledgements
We thank the [RePo](https://github.com/zchuning/repo) authors for their implementation of Pytorch version of Dreamer and Distracted DMC wrappers.