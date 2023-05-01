import collections

import numpy as np


class ReplayMemory:
    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are
        # written into the memory from the randomly initialized agent.
        # Memory size is the maximum size after which old elements in
        # the memory are replaced. A simple (if not the most efficient) was
        # to implement the memory is as a list of transitions.
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions -
        # i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        memory_len = len(self.memory)
        indices = np.random.choice(range(memory_len), size=batch_size)
        samples = []
        for i in indices:
            samples.append(self.memory[i])
        return samples

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)
