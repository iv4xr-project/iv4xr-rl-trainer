# iv4XR RL trainer

This repository contains code to connect to Reinforcement Learning environments 
run by the iv4XR framework and train a Reinforcement Learning agent.

## Installation

The `iv4xrl` folder is a small library that manages the interoperability with the
iv4XR framework. It is the Python counterpart connector and Gym environment to
the iv4xr-rl-env JAVA project.
The `iv4xrl` library can be installed locally by running the following command in
its folder.
```
pip install -e .
```

## Usage

The `test_agent.py` script can be run for simple tests of random or deterministic
agent on the iv4XR-controlled environment.
You will need to run the associated JAVA test `RLAgentConnectTest` found in the
iv4xr-rl-env project.

## Code architecture

The `iv4xrl` library manages the usage of iv4XR-based RL Environments as common
Gym environments.

The `trainings` folder contains the implementation of the Deep Reinforcement Learning
algorithms and their adaptation to the Maze environment, a simplified setup of the
powerplant intrusion simulation that shares the same interfaces and logic.

- The TD3 [0] algorithm is used and adapted as the goal-solving Functional Test Agent (FTA)
  for the intrusion use-case.
- The QD-RL [1] algorithm is used and adapted as the behavioural coverage FTA for the
  intrusion use-case.

## Approach and results

The approach and results are detailed in this project's Wiki:
- https://github.com/iv4xr-project/iv4xr-rl-trainer/wiki/Approach-and-Results---Goal-Solving-FTA
- https://github.com/iv4xr-project/iv4xr-rl-trainer/wiki/Approach-and-Results---Behavioural-Coverage-FTA

## References

[0] Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." International Conference on Machine Learning. PMLR, 2018.

[1] Cideron, Geoffrey, Thomas Pierrot, Nicolas Perrin, Karim Beguir, and Olivier Sigaud. "QD-RL: Efficient Mixing of Quality and Diversity in Reinforcement Learning." arXiv preprint arXiv:2006.08505 (2020)
