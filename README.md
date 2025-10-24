# UAM_v2

## Overview

As urban air mobility (UAM) slowly becomes a reality, the strategic placement of vertiports (takeoff and landing areas for vertical takeoff and landing (VTOL) aircraft) will become a crucial design problem to ensure safe and efficient travel while limiting audial impact on surrounding areas. This work aims to use GNNs-RL to determine the optimal placement of these vertiports. By using RL we aim to... finish this with the design constraints of the RL algorithm.

## Setup

### Getting Started (ubuntu):

```
git clone https://github.com/chief-khanman/UAM_v2
cd UAM_v2
conda config --add channels conda-forge
conda env create -f environment_ubuntu.yml
conda activate AAM_AMOD
```

You will want to activate this conda environment anytime you want to run anything.

If you are using VScode set your environment in the bottom right of your screen to make running files easier

## Executables

main.py: Runs simulation with basic UAV's

#### Urban Air Mobility env

This is a custom gym environment, that has been developed for training and testing single agent.
The single agent is defined using the auto_uav variable in this script.

##### Description of the environment:

The objective of the environment is to create an airspace where our autonomous uav will 'fly'.
Autonomous UAV will be provided a start and end point, called vertiports, its objective is to traverse from its start vertiport to its end vertiport.
While traversing, it has to avoid 'restricted airspace' and other UAVs.

We define an airspace using a location name. Then we sample vertiports within from our airspace.
After, we create UAVs which fly using hardcoded policy. An instance of autonomous uav is deployed.

> [!NOTE]
> If anyone wishes to improve/change the environment I suggest once the assets have been created/updated test them with `main.py`. There are `simulator/simulator_basic.py` scripts which define an environment with all necessary assets.
>
> Any improvement/change should be tested using instance of simulator before implementing in custom gym env.
>
> When we create the UAM env(subclass of gymEnv) it will build an instance that is similar to simulator. The initializer arguments of UAM_Env are similar to the simulator, that is location_name, basic_uav_no, vertiport_no, and Auto_uav(only one for single agent env)
>
> ### Reason for similarity
>
> The simulator scripts create environments where we test functionality of assets. All assets are pulled in to create an environment similar to this one, only auto_uav is not part of simulator.
>
> Any form of changes to assets should first be tested by using instance of simulator, then added accordingly to this environment. This process will make debugging easier.

### Training

The following scripts are used for training the agent(s)

#### Single Agent

`train_single_agent.py`: Runs simulation for with one agent and specified number of basic uav's

Use this script to train a model using the custom UAM environment.
Before training, make sure to run check_uam.py to ensure the environment is in working condition.
This check must be done anytime there is a change to the environment.

Once the check is complete you can proceed with training a model using algorithms from sb3

```
conda install conda-forge::stable-baselines3
```

#### Multi Agent

`train_multi_agent.py`: All UAVs are learning agents

May be required if this was missed in conda env create:

```
pip install pettingzoo
```
