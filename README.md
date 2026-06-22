# Module 1: Deep Reinforcement Learning for Traffic Signal Control

This module implements a Deep Q-Network-based adaptive traffic signal controller using SUMO, TraCI, Python, and PyTorch. The goal is to show how reinforcement learning can be used to control traffic signal phases based on real-time traffic conditions in a simulated intersection.

## Overview

In traditional traffic signal control, signal timings are often fixed or rule-based. In this module, a reinforcement learning agent learns how to select signal phases by interacting with the SUMO traffic simulation environment.

The agent observes the traffic state, chooses an action, receives a reward based on traffic performance, and gradually improves its decision-making policy over multiple training episodes.

## Main Components

This module includes:

* SUMO network files
* Route and traffic demand files
* SUMO configuration file
* Custom traffic signal environment using TraCI
* Deep Q-Network controller implemented in PyTorch
* Training and evaluation scripts

## Repository Structure

```text
Module_1_DeepRLControl/
│
├── nodes.xml
├── edges.xml
├── routes.xml
├── sumocfg.xml
├── gui_settings.xml
├── sumo_env.py
├── dqn_controller.py
└── RUN.sh
```

## Objective

The main objective of this module is to train an RL-based traffic signal controller that can:

* Observe traffic conditions at an intersection
* Select appropriate signal phases
* Reduce vehicle waiting time
* Improve traffic flow
* Learn better control policies through repeated simulation episodes

## Reinforcement Learning Setup

The RL setup contains the following basic elements:

### State

The state represents the current traffic condition observed from the SUMO simulation. This can include information such as vehicle queues, lane occupancy, waiting time, or traffic density near the intersection.

### Action

The action represents the traffic signal decision selected by the RL agent. For example, the agent may choose which signal phase should be active at the intersection.

### Reward

The reward tells the agent whether its action was good or bad. A common reward is based on reducing vehicle waiting time, queue length, or congestion. The agent tries to maximize the total reward over time.

### Agent

The agent is a Deep Q-Network model that learns the best action to take for each traffic state. It uses past experiences to improve its signal control strategy.

## Requirements

Make sure the following tools and libraries are installed:

* Python 3.8 or higher
* SUMO
* TraCI
* PyTorch
* NumPy

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

Move into the Module 1 folder:

```bash
cd Module_1_DeepRLControl
```

Generate the SUMO network file:

```bash
netconvert -n nodes.xml -e edges.xml -o intersection.net.xml
```

Train the DQN traffic signal controller:

```bash
python dqn_controller.py --train --episodes 20
```

Evaluate the trained controller with SUMO GUI:

```bash
python dqn_controller.py --eval --model dqn_model.pth --gui
```

## Expected Output

After training, the model learns a traffic signal control policy. During evaluation, the trained agent controls the intersection in SUMO and selects signal phases based on the observed traffic state.

The output may include:

* Training reward values
* Episode-level performance
* Trained model file
* SUMO GUI visualization during evaluation


