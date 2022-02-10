import os
import pickle

import torch.cuda
from from_root import from_root
from torch.utils.data import DataLoader

from agents.agent_alphafour.NN import Alpha_Net, AlphaLossFunction


def load_NN(NN, NN_iteration):
    NN_filename = from_root(f"agents/agent_alphafour/trained_NN/NN_iteration{NN_iteration}.pth.tar")
    start_epoch = 0
    loaded_NN = None
    if os.path.isfile(NN_filename):
        loaded_NN = torch.load(NN_filename)
    if loaded_NN is not None:
        NN.load_state_dict(loaded_NN["state_dict"])
    return start_epoch


def train(NN, dataset, optimizer, scheduler, num_of_epochs=300):
    torch.manual_seed(0)
    # Turn on training mode
    NN.train()
    # Choose criteria
    criteria = AlphaLossFunction()
    # Load train set
    training_loader = DataLoader()
    for epoch in range(0, num_of_epochs):
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(training_loader, 0):
            state, policy, value = data
            # Feed the NN
            policy_prediction, value_prediction = NN(state)
            loss = criteria()
        scheduler.step()
        # TODO: Do more


def trainNN(NN_iteration, learning_rate=0.001, ):
    # Load saved data
    dataset = []
    data_path = "agents/agent_alphafour/training_data/"
    for i, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as f:
            dataset.extend(pickle.load(f, encoding='bytes'))
    # Train the NN
    NN = Alpha_Net()
    print("Training...")
    # Turn on CUDA if avaiable
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    NN.to(dev)
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate, betas=(0.8, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300, 400],
                                                     gamma=0.77)
    # Load state
    load_NN(NN, NN_iteration)
    # Train
    train(NN, dataset, optimizer, scheduler)

