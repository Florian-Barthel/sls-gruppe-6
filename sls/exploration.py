import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def epsilon_greedy(actions, q_table_row, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.choice(actions)
    else:
        return actions[np.argmax(q_table_row)]


def boltzmann(actions, q_table_row, temperature):
    probabilities = softmax(q_table_row / temperature)
    return np.random.choice(actions, p=probabilities)
