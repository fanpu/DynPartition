# DynPartition
Code for the paper [DynPartition: Automatic Optimal Pipeline Parallelism of Dynamic Neural
Networks over Heterogeneous GPU Systems for Inference Tasks](paper.pdf).

The DynPartition framework is built on top of PyTorch, it uses a reinforcement learning
scheduler based on Deep-Q Learning for generating an optimal allocation policy
across multiple GPUs for Tree-typed dynamic neural networks. The framework
supports both static and dynamic partitioning of the network.
The framework is designed to be modular and can be easily extended to support
other types of neural networks and other types of partitioning algorithms.

## Installation

The framework is built on top of PyTorch, so you need to install PyTorch first.
then you can install the framework using pip:

```
pip install -r requirements.txt
```

## Usage

Run all the cases using the following command:

```
cd DynPartition
export PYTHONPATH="${pwd}"
sh dynPartition/run.sh
```

The results will be saved in the `dynPartition/_logs`
directories.

## Results

The results can be created by using scripts in the `dynPartition/graphs`
directory.

## Report

The detailed analysis of this framework is presented in [our paper](paper.pdf).

## Branches

The framework is implemented in two branches:

- `master`: uses Reinforcement Learning scheduler based on Deep-Q Learning.
- `using_policy_value`: uses Reinforcement Learning scheduler based on Policy
  Gradient.
