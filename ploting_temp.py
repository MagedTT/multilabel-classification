import matplotlib.pyplot as plt
import torch


def plot_(logging_dict1, logging_dict2, mode='accuracy'):
    if mode == 'accuracy':
        plot_accuracy(logging_dict1, logging_dict2)

    elif mode == 'loss':
        plot_loss(logging_dict1, logging_dict2)

    else:
        raise ValueError('Unexpected model value. Available values are `accuracy` or `loss`.') 



def plot_accuracy(logging_dict1, logging_dict2):
    plt.plot(range(len(logging_dict1)), logging_dict1, label='Train')
    plt.plot(range(len(logging_dict2)), logging_dict2, label='Validation')
    plt.title('Train-Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()


def plot_loss(logging_dict1, logging_dict2):
    plt.plot(range(len(logging_dict1)), logging_dict1, label='Train')
    plt.plot(range(len(logging_dict2)), logging_dict2, label='Validation')
    plt.title('Train-Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.show()


