# Human Model Validation
This python module contains scripts needed to train IRL Driver models on HighD datasets. This code accompanies the paper "Validating human driver models for
interaction-aware automated vehicle controllers: A human factors approach - Siebinga, Zgonnikov & Abbink 2021". A pre-print version of this paper is 
available on [arXiv](https://arxiv.org/abs/2109.13077).

## Installation instructions
This module cannot be used on its own, its should be used as a sub-module of the TraViA visualization software (click 
[here](https://joss.theoj.org/papers/10.21105/joss.03607) for the paper, or [here](https://github.com/tud-hri/travia) for the repository). If you want to 
use this module, first make sure you have a working version of TraViA. You can clone it directly from github using the following command, or fork it first 
and clone your own version.

```
git clone https://github.com/tud-hri/travia.git
```

After cloning TraViA, you can navigate to the travia folder (`cd travia`) and clone this repository as a submodule. Use the following command to clone the 
girhub version, or create a fork first and then clone your own fork.

```
git submodule add https://github.com/tud-hri/irlmodelvalidation.git
```

Instruction on how to get the data and how to work with TraViA can be found in the TraViA README file. See the instructions below for how to work with this 
sub-module.

## General overview 
An object representing the IRL agent can be constructed using the `IRLAgent` class from `irlagent.py`. The script `evaluatemodel.py` uses an agent object
to calculate the response of this agent in a specific dataset. The script `irlagenttools.py` contains helper functions for the IRL agent. Training a set of 
agents can be done by running `train_irl.py`. This script uses `rewardgradient.py` to find the gradient and Hessian of the reward function.

## Training agents
The script `train_irl.py` can be used to train irl agents on demonstrations from the HighD dataset. These demonstrations are automatically selected from the 
dataset. Each successful IRL training results in a set of weight. These weights are used in an agent and the agent's behavior is simulated. The results of 
this simulation are stored in a pickle file for every agent. The main block of the script is set up to train on all demonstrations in a single dataset. It 
uses multiprocessing to train agents in parallel. Please note that training all on all demonstrations of a single dataset can take a long time. The main 
block can be altered to train on fewer demonstrations or with other parameters. Alternatively, the `fit_theta_and_simulate` function can be used directly to 
train with other parallelization then multiprocessing, e.g. on a cluster.  

Please be aware that the training function uses the pickled version of the dataset.This version is automatically saved if the dataset was visualized at 
least once.

If the `gridsearch` boolean is set to `True`, the training is done in the same way but the file is saved in a different location. The file name is also 
appended with the parameter values to make sure multiple version of the same agent id can be saved. Grid search results can be evaluated with the 
`evaluate_grid_search.py` script.  

## Evaluating the results
The evaluation of the driver model takes place in two stages (for more information see the accompanying paper). The first stage is the tactical validation, 
this can be done using the script `tacital_evaluation.py`. This script will automatically find all saved agent files in the data folder and summarize the 
tactical results in a human-readable text file. 

Before evaluating the operational behavior of the trained agents, the metrics that are needed for that evaluation need to be calculated. This is done by running
the script `compute_collision_metrics.py`.

After running that the script `operational_validation.py` can be used for the operational evaluation of the results. This script will produce the plots as
can be found in the paper. This is again based on all agent files in the data folder. Please be aware that these scripts should be run in this order because 
the tactical results are needed for the operational evaluation.

## Visualizing the results
To visualize the results of a trained agent, run the script `visualize_agent.py`. In this file you can specify a dataset id and the vehicle ID of the 
trained IRL agent.  

## Other Scripts and Files
`compute_data_statistics.py` can be used to calculate some statistics as a summary of the HighD dataset. These statistics were reported in the paper.

`find_carfollowing_examples.py` contains a helper function that is used to plot the human car following behavior in the operational results plot.

`irlagenttools.py` contains a helper function for the irl agent that is also used in places were the agent itself is not used, hence the separate file.

`strategicbehavior.py` contains the enum specifying the different types of regarded tactical behaviors.