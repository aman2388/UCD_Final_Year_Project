import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset, DataLoader

from model.viterbi import v_fViterbi


class CnnViterbiNet(nn.Module):
    def __init__(self):
        super(CnnViterbiNet, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=4)
        self.conv1d2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=4)
        self.conv1d3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=4)
        self.maxpool = nn.MaxPool1d(2)
        # 128x16
        self.fc = nn.Linear(128, 16)

    def forward(self, x):
        x = x.unsqueeze(1)
        # layer 1
        x = self.conv1d(x)
        # relu so that all values are positive
        x = F.relu(x)
        # take the top 2 values of the input after applying relu function
        x = self.maxpool(x)
        # layer2
        x = self.conv1d2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # layer3
        x = self.conv1d3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # flatten
        x = x.squeeze()

        # feed to a fully-connected layer
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

    def TrainViterbiNet(self, channel_train, symbol_train, const_size):
        # -----Training-----#
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        epochs = 100
        miniBatchSize = 64

        train_length = len(symbol_train)  # train length = 4 (4x5000)

        # combine each set of symbols as a single unique category
        combine_vec = const_size ** np.array(
            [np.arange(train_length)])  # 2^n i.e. 2^1, ... 2^4. Results looks like this: np.array([[1, 2, 4, 8]])

        # multiply each category to get 1x5000 symbols (16 unique symbols)
        symbol_train = combine_vec.dot(symbol_train - 1)

        channel_train = channel_train.reshape(channel_train.shape[1], channel_train.shape[0])
        symbol_train = symbol_train.reshape(symbol_train.shape[1], symbol_train.shape[0])

        # one-hot-encode the numpy array as there are 16 classes
        symbol_train = np.eye(16)[symbol_train.reshape(-1).astype(int)]

        # convert the np array to tensors
        symbol_train = torch.from_numpy(symbol_train).float()
        channel_train = torch.from_numpy(channel_train).float()

        # combine tensors into a tensor-dataset so that we can apply batches in dataloader
        train_set = TensorDataset(channel_train, symbol_train)
        loader = DataLoader(train_set, batch_size=miniBatchSize, pin_memory=True, shuffle=True)
        running_loss = np.zeros(epochs)

        # start training
        for epoch in range(epochs):
            for train_channel, train_symbol in loader:
                self.zero_grad()
                optimizer.zero_grad()  # calculate the gradient, manually setting to 0

                batch_outputs = self.forward(train_channel).float()  # forward pass
                # obtain the loss function
                loss = criterion(batch_outputs, train_symbol)

                loss.backward()  # calculates the loss of the loss function
                optimizer.step()  # improve from loss, i.e backprop

                running_loss[epoch] += loss.item()
            scheduler.step()
        # print
        print('-training-')
        print(f'CNN Network Loss: {running_loss.mean()}')

    def TestViterbiNet(self, channel_train, train_size, channel_test, test_size, const_size, memory_length):

        """
        Apply ViterbiNet on unseen channel outputs
        -------------------------------------------------------
        INPUT:
        -------------------------------------------------------
        channel_train - training channel outputs to recover symbols
        train_size - 5000
        channel_test - unseen channel outputs
        test_size - 50000
        const_size - constellation size
        memory_length - channel size (4)
        OUTPUT:
        -------------------------------------------------------
        predicted symbols after Viterbi decoder layer
        """

        num_states = pow(const_size, memory_length)
        # implement the expectation-maximization algorithm for fitting mixture of gaussian models
        GMModel = GaussianMixture(n_components=num_states, max_iter=100).fit(channel_train.reshape(train_size, 1))

        # compute the log-likelihood of each sample
        out_pdf = -GMModel.score_samples(channel_test.reshape(test_size, 1))

        # change numpy array to tensor
        channel_test = channel_test.reshape(channel_test.shape[1], channel_test.shape[0])
        channel_test = torch.from_numpy(channel_test).float()

        # reshaping to rows, timestamps, features
        # channel_test_final = torch.reshape(channel_test,(channel_test.shape[0], 1, channel_test.shape[1]))

        # predict symbols
        test_net = self.forward(channel_test.float())
        # reshape the predicted outputs to 50000x16
        nn_output = np.reshape(test_net.detach().numpy(), newshape=(test_size, num_states))
        # compute log-likelihoods
        m_fLikelihood = np.multiply(nn_output, out_pdf.reshape(test_size, 1)) / num_states

        # Apply Viterbi output layer based on the DNN likelihood
        v_fXhat = v_fViterbi(m_fLikelihood, const_size, memory_length)
        return v_fXhat
