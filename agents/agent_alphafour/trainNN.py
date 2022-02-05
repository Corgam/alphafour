import os
import pickle

from agents.agent_alphafour.NN import AlphaNet


def trainNN():
    # Load saved data
    dataset = []
    data_path = "agents/agent_alphafour/training_data/"
    for i, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as f:
            dataset.extend(pickle.load(f, encoding='bytes'))
    # Train the NN
    NN = AlphaNet()
    print("Training...")