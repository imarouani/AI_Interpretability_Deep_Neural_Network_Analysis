import numpy as np

def generate_sample_data(dist_list, n=300, d=3, p=[1/3]*3):
    '''
    Generates random labeled data that can then be used for clustering.
    
    :param dist_list (list): List of distributions for each class in form of a list of tuples (x_mean, y_mean, x_std, y_std).
    :param n (int): Number of total samples to be returned.
    :param d (int): Number of classes present in the data.
    :param p (int): Proportion of total samples allocated to each class.

    :returns (tuple): Two-dimensional data and a label array in shapes (n, 2), (n, 1).
    '''

    if len(dist_list) != d or len(p) != d:
        raise ValueError('Number of classes {d} does not match the provided distributions {len(dist_list)} or proportions {len(p)}')

    data = []
    y = []
    nums = [int(n*prop) for prop in p]
    if sum(nums) != n:
        nums[np.argmax(p)] += 1
    
    for i, params in enumerate(dist_list):
        x_mean, y_mean, x_std, y_std = params
        data += [np.hstack([np.random.normal(x_mean, x_std, (nums[i], 1)), np.random.normal(y_mean, y_std, (nums[i], 1))])]
        y += [np.array([i+1]*nums[i]).reshape((nums[i], 1))]

    return np.vstack(data), np.vstack(y)