import numpy as np
import pickle

import os


def load_and_prepare_data(root_path, as_grayscale=False):
    """Load raw data using pickle."""
    # Check if the cifar10 dataset has been downloaded
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'

    # Load the training data
    try:
        for batch in train_batches:

            batch = os.path.join(root_path, batch)
            print(batch)

            with open(batch, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                data = batch[b'data']
                labels = batch[b'labels']
                if 'train_data' in locals():
                    train_data = np.concatenate((train_data, data))
                    train_labels = np.concatenate((train_labels, labels))
                else:
                    train_data = data
                    train_labels = labels
    except FileNotFoundError:
        print("The CIFAR-10 dataset has not been downloaded. Download and extract the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place the data_batch files in the cifar10 directory.")
        return
    
    # Load the test data
    try:

        test_batch = os.path.join(root_path, test_batch)

        with open(test_batch, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
            test_data = batch[b'data']
            test_labels = batch[b'labels']
    except FileNotFoundError:
        print("The CIFAR-10 dataset has not been downloaded. Download and extract the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place the data_batch files in the cifar10 directory.")
        return
    
    # Reshape the data
    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)

    if as_grayscale:
        train_data = np.dot(train_data, [0.299, 0.587, 0.114])
        test_data = np.dot(test_data, [0.299, 0.587, 0.114])

    return train_data, train_labels, test_data, test_labels

