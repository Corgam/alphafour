import os
import pickle

import numpy as np
import torch
from from_root import from_root
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_

from agents.agent_alphafour.NN import AlphaNet, AlphaLossFunction


class BoardDataset(Dataset):
    """
    Dataset for storing triplets of the board, policies and values.
    """
    def __init__(self, dataset: list):
        self.boards = [data[0] for data in dataset]
        self.policies = [data[1] for data in dataset]
        self.values = [data[2] for data in dataset]

    def __len__(self):
        """
        Total length of the dataset.
        :return: length
        """
        return len(self.boards)

    def __getitem__(self, idx: int):
        """
        Returns a specific item from given index.
        :param idx: index
        :return: item_with_specified_index
        """
        return np.int64(self.boards[idx]), self.policies[idx], self.values[idx]


def load_nn(nn: AlphaNet, iteration: int):
    """
    Loads the saved state of the NN into the NN object.
    :param nn: the NN object to load into
    :param iteration: the iteration of NN to load from
    :return:
    """
    nn_filename = from_root(
        f"agents/agent_alphafour/trained_NN/NN_iteration{iteration}.pth.tar"
    )
    loaded_nn = None
    if os.path.isfile(nn_filename):
        loaded_nn = torch.load(nn_filename)
    if loaded_nn is not None:
        nn.load_state_dict(loaded_nn["state_dict"])


def train(nn: AlphaNet, dataset: list, optimizer: Adam, scheduler: MultiStepLR, num_of_epochs: int, iteration: int):
    """
    Inner function for training the NN, requires optimizer and scheduler to be passed in.
    Do not call directly.
    :param nn: the NN to train
    :param dataset: the dataset to train from
    :param optimizer: the optimizer used in training
    :param scheduler: the scheduler used in training
    :param num_of_epochs: number of epochs to run
    :param iteration: current iteration of the main pipeline
    """
    torch.manual_seed(0)
    # Turn on training mode
    nn.train()
    # Choose criteria
    criteria = AlphaLossFunction()
    # Load train set
    training_set = BoardDataset(dataset)
    training_loader = DataLoader(training_set, batch_size=1, shuffle=True)
    for epoch in range(0, num_of_epochs):
        for i, data in enumerate(training_loader, 0):
            state, policy, value = data
            state = state.float()
            value = value.float()
            # Feed the NN
            state = np.expand_dims(state, 1)
            state = torch.from_numpy(state)
            policy_prediction, value_prediction = nn(state)
            # Calculate the loss
            loss = criteria(value_prediction[:, 0], value)
            loss.backward()
            clip_grad_norm_(nn.parameters(), 1.0)
            # Forward the optimizer
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        # Save the NN
        nn_filename = from_root(
            f"agents/agent_alphafour/trained_NN/NN_iteration{iteration + 1}.pth.tar"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": nn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            nn_filename,
        )


def train_nn(iteration: int, num_of_epochs: int, learning_rate: float = 0.001):
    """
    Main function for training the NN.
    Loads the NN and creates the optimizer and scheduler.
    :param iteration: starting iteration of the NN to load
    :param num_of_epochs: how many epochs the training will take
    :param learning_rate: the learning rate of the NN
    """
    print("[TRAINING] Started training!")
    # Load saved data
    dataset = []
    data_path = f"agents/agent_alphafour/training_data/iteration{iteration}/"
    for i, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            dataset.extend(data)
    nn = AlphaNet()
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, betas=(0.8, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150, 200, 250, 300, 400], gamma=0.77
    )
    # Load the state
    load_nn(nn, iteration)
    # Train the NN
    train(nn, dataset, optimizer, scheduler, num_of_epochs, iteration)
    print("[TRAINING] Finished training!")
