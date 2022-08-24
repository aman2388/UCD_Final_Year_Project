import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from model.viterbi import v_fViterbi


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim),self.weight)
            #torch.bmm(x, self.weight.unsqueeze(0).repeat(5000, 1, 1))

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class AttentionLSTM(nn.Module):
    def __init__(self, hidden_dim=100, lstm_layer=1, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=100,
                             num_layers=1, batch_first=True)

        self.atten1 = Attention(hidden_dim, 100)  # 2 is bidrectional
        self.atten2 = Attention(hidden_dim, 100)  # 2 is bidrectional
        self.atten3 = Attention(hidden_dim, 100)  # 2 is bidrectional
        self.embedding = nn.Embedding(16,5000)

        self.fc1 = nn.Linear(in_features=100, out_features=16)

    def forward(self, x):

        if torch.cuda.is_available():
            h_0 = Variable(torch.zeros(1, x.size(0), 100).cuda())
        else:
            h_0 = Variable(torch.zeros(1, x.size(0), 100))

        if torch.cuda.is_available():
            c_0 = Variable(torch.zeros(1, x.size(0), 100).cuda())
        else:
            c_0 = Variable(torch.zeros(1, x.size(0), 100))



        out1, (hn, cn) = self.lstm1(x, (h_0, c_0))
        ans1 = self.atten1(out1)  # skip connect
        ans2 = self.atten2(out1)
        ans3 = self.atten3(out1)
        avg_pool = torch.mean(out1, 1)
        max_pool, _ = torch.max(out1, 1)
        z = ans1 + ans2 + ans3 + max_pool + avg_pool

        out = self.fc1(z)
        out = torch.sigmoid(out)

        return out


    def TrainViterbiNet(self, channel_train, symbol_train, const_size):
        np.random.seed(9001)
        """
        Train ViterbiNet conditional distribution network
        Syntax
        -------------------------------------------------------
        channel_train - training channel outputs to recover symbols
        symbol_train - training symobls corresponding to each channel output
        const_size - constellation size
        learn_rate - learning rate (0.0005)
        
        OUTPUT:
        -------------------------------------------------------
        trained neural network model
        """

        # -----Training-----#
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
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
        print(f'Attention LSTM Loss: {running_loss.mean()}')

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

# net = MyLSTM()
#
# bsize = 27
# inp = torch.randn((bsize, 1))
#
# inp = torch.reshape(inp, (inp.shape[0], 1, inp.shape[1]))
#
# out = net(inp)
# print(out.shape)
