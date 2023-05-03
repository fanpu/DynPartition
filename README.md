# DynPartition
DynPartition framework built on top of PyTorch, it uses Reinforcement Learning
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
The results will be saved in the `dynPartition/_plots` directory.

## Results
The detailed analysis of this framework is presented in the "Project Report.pdf"
file.

