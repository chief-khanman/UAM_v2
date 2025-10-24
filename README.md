# UAM_v2

## Overview

As urban air mobility (UAM) slowly becomes a reality, the strategic placement of vertiports (takeoff and landing areas for vertical takeoff and landing (VTOL) aircraft) will become a crucial design problem to ensure safe and eficient travel while limiting audial impact on surrounding areas. This work aims to use GNNS-RL to determine the optimal placement of these vertiports. By using RL we aim to... finish this with the design constraints of the RL algorithm.

## Setup


### Getting Started (ubuntu):

```
mkdir folder_name
```

```
cd folder_name
```

```
git clone https://github.com/chief-khanman/UAM_v2.git
```

```
cd folder_name/UAM_v2
```

```
conda config --add channels conda-forge
```

```
conda env create -f environment_ubuntu.yml
```

```
conda activate AAM_AMOD
```

You will want to activate this conda environment anytime you want to run somthing.

If you are using VScode set your environment in the bottoom right of your screen to make running files easier

### Setting up RVO2 
 
After completing the steps above for creating the conda environment, follow the steps in Python-RVO2 README to build and install Python-RVO2. Make sure you are using - pip install Cython

This command installs the latest Cython which will enable you to install pyrvo2 in your conda environment without any issues.

## Exacutables

main.py: Runs simulation with basic UAV's

### Single Agent

train_single_agent.py: Runs simulation for with one agent and specified number of basic uav's

May be required if this was missed in conda env create:

```
conda install conda-forge::stable-baselines3
```

### Multi Agent

train_multi_agent.py: All UAVs are learning agents

May be required if this was missed in conda env create:

```
pip install pettingzoo

```
##### From aw in lab

