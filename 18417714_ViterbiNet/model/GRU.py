import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from model.viterbi import v_fViterbi


class multiGRU(nn.Module):
    def __init__(self, input_size, hidden_unit_size, num_layers, class_size):
        super(multiGRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_unit_size = hidden_unit_size

        # set the ViterbiNet architecture.
        # 1 x 100 , 100 x 100, 100 x 16

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_unit_size,
                          num_layers=num_layers, batch_first=True)

        self.gru2 = nn.GRU(input_size=input_size, hidden_size=hidden_unit_size,
                           num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_unit_size, out_features=class_size)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_unit_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_unit_size))

        output, h1 = self.gru(x, h0.detach())
        output2, h2 = self.gru2(x, h1.detach())
        out = output2[:, -1, :]

        out = self.fc(out)
        #out = F.softmax(out, dim=1)
        out = torch.sigmoid(out)
        return out

    def TrainViterbiNet(self, channel_train, symbol_train, const_size):
        np.random.seed(9001)
        """
        Train ViterbiNet conditional distribution network
        Syntax
        -------------------------------------------------------
        net = TrainViterbiNet(X_train,Y_train ,s_nConst, layers, learnRate)
        INPUT:
        -------------------------------------------------------
        channel_train - training channel outputs to recover symbols
        symbol_train - training symobls corresponding to each channel output
        const_size - constellation size
        learn_rate - learning rate (0.00005)
        OUTPUT:
        -------------------------------------------------------
        trained neural network model
        """

        # -----Training-----#
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
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

                # reshaping to rows, timestamps, features
                train_channel_final = torch.reshape(train_channel,
                                                    (train_channel.shape[0], 1, train_channel.shape[1]))
                batch_outputs = self.forward(train_channel_final).float()  # forward pass
                # obtain the loss function
                loss = criterion(batch_outputs, train_symbol)

                loss.backward()  # calculates the loss of the loss function
                optimizer.step()  # improve from loss, i.e backprop

                running_loss[epoch] += loss.item()
            scheduler.step()
        # print
        print('-training-')
        print(f'GRU Loss: {running_loss.mean()}')

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
        channel_test_final = torch.reshape(channel_test,
                                           (channel_test.shape[0], 1, channel_test.shape[1]))

        # predict symbols
        test_net = self.forward(channel_test_final.float())
        # reshape the predicted outputs to 50000x16
        nn_output = np.reshape(test_net.detach().numpy(), newshape=(test_size, num_states))
        # compute log-likelihoods
        m_fLikelihood = np.multiply(nn_output, out_pdf.reshape(test_size, 1)) / num_states

        # Apply Viterbi output layer based on the DNN likelihood
        v_fXhat = v_fViterbi(m_fLikelihood, const_size, memory_length)
        return v_fXhat
