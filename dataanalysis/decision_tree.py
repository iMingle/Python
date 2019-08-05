"""决策树

"""
import numpy as np


def entropy(data_df):
    x_value_list = set([data_df[i] for i in range(data_df.shape[0])])
    entropy_result = 0.0
    for x_value in x_value_list:
        p = float(data_df[data_df == x_value].shape[0]) / data_df.shape[0]
        logp = np.log2(p)
        entropy_result -= p * logp

    return entropy_result


def entropy_prob(prob):
    entropy_result = 0.0
    for value in prob:
        logp = np.log2(value)
        entropy_result -= value * logp

    return entropy_result


if __name__ == '__main__':
    prob = np.array([1 / 6, 5 / 6])
    print(entropy_prob(prob))
    data = np.array([1, 2, 2, 2, 2, 2])
    print(entropy(data))
