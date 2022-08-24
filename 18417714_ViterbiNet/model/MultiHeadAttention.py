import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import math

from model.viterbi import v_fViterbi

##Taken from - https://github.com/hyunwoongko/transformer

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=200, n_head=5):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(1, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        #self.w_k = nn.Linear(d_model, d_model)
        #elf.w_v = nn.Linear(d_model, d_model)
        #self.w_concat = nn.Linear(d_model, 16)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 16)

    def forward(self, q, mask=None):
        # 1. dot product with weight matrices
        q = self.w_q(q)
        q = F.relu(q)
        q = self.w_k(q)
        q = F.relu(q)
        q = self.w_v(q)
        q = F.relu(q)

        # 2. split tensor by number of heads
        q = self.split(q)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q,q,q, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = out.view(-1, 200)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    # def forward(self, x):
    #     attn_output, attn_output_weights = self.mha(x, x, x)
    #     x = self.flat(attn_output)
    #     x = x.view(-1, 200)
    #     x = self.fc(x)
    #     x = torch.sigmoid(x)
    #
    #     return x


    # def forward2(self, x):
    #     h_0 = Variable(torch.zeros(2, x.size(0), 100))
    #     c_0 = Variable(torch.zeros(2, x.size(0), 100))
    #     out1, (hn, cn) = self.lstm(x, (h_0, c_0))
    #     attn_output, attn_output_weights = self.mha(out1, out1, out1)
    #     x = self.flat(attn_output)
    #     x = x.view(-1, 200)
    #     x = self.fc(x)
    #     x = torch.sigmoid(x)
    #
    #     return x

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

        #-----Training-----#
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
        print(f'MultiHeadAttention Loss: {running_loss.mean()}')

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
        channel_test_final = torch.reshape(channel_test,(channel_test.shape[0], 1, channel_test.shape[1]))

        # predict symbols
        test_net = self.forward(channel_test_final.float())
        # reshape the predicted outputs to 50000x16
        nn_output = np.reshape(test_net.detach().numpy(), newshape=(test_size, num_states))
        # compute log-likelihoods
        m_fLikelihood = np.multiply(nn_output, out_pdf.reshape(test_size, 1)) / num_states

        # Apply Viterbi output layer based on the DNN likelihood
        v_fXhat = v_fViterbi(m_fLikelihood, const_size, memory_length)
        return v_fXhat