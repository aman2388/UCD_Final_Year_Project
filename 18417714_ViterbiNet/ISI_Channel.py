import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from model.CNN import CnnViterbiNet
from model.MultiHeadAttention import MultiHeadAttention
from model.multi_lstm import multiLSTM
from model.GRU import multiGRU
from matrix_reshape import matrix_reshape
from model.viterbi import v_fViterbi
from model.viterbinet import viterbinet
import torch

np.random.seed(9001)

const_size = 2  # Constellation size (2 = BPSK)
channel_len = 4  # channel size
train_size = 5000  # Training size
test_size = 50000  # Test data size

num_states = pow(const_size, channel_len)

v_fSigWdB = np.arange(-6, 10 + 1, 2)

error_var_est = 0.1  # Estimation error variance

# Frame size for generating noisy training
frame_size = 500

# split the train data size into 10 subsets
partition_size = int(train_size / frame_size)

# curve index for semi-logy graph
model_curves = [
    1,  # ViterbiNet - perfect CSI
    1,  # ViterbiNet - CSI uncertainty
    1,  # Viterbi algorithm - perfect CSI
    1,  # Viterbi algortihm - estimated CSI
]

curve_size = len(model_curves)

m_reshape = matrix_reshape()

fading_exp = np.array([0.2])

# a 4x9 matrix containing 9 symbol error estimates for 4 models
m_fSER = np.zeros((np.size(model_curves), np.size(v_fSigWdB)))

# e^-(n-1)/5,
# where n = channel index
fading_channel = np.array([np.exp(-fading_exp * np.arange(0, channel_len, 1))])

# train label
train_symbol = np.array([np.random.randint(1, const_size + 1, train_size)])
train_symbol_reshape = m_reshape.reshape_data(train_symbol, channel_len)
# test label
test_symbol = np.array([np.random.randint(1, const_size + 1, test_size)])

# convert vector values in a BPSK constellation (-1,1)
train_bpsk = 2 * (train_symbol - 0.5 * (const_size + 1))
test_bpsk = 2 * (test_symbol - 0.5 * (const_size + 1))

# reshape train vector into a 4x5000 matrix
train_bpsk_reshaped = m_reshape.reshape_data(train_bpsk, channel_len)

# Multiply matrices with array from reversed exp_vector (ht) and reshaped array
train_bpsk_product = np.matmul(np.fliplr(fading_channel), train_bpsk_reshaped)

# reshape test vector into a 4x50000 matrix
test_bpsk_reshaped = m_reshape.reshape_data(test_bpsk, channel_len)
# Multiply matrices with array from exp_vector (reversed) and reshaped array
test_bpsk_product = np.matmul(np.fliplr(fading_channel), test_bpsk_reshaped)

# noisy csi
train_noise = np.array([np.zeros((np.size(train_bpsk_product)))])

# To compute noise, we must ensure that the receiver only has access to a noisy estimate of the fading channel and
# specifically, to a copy of fading channel whose entries are corrupted by
# i.i.d. zero-mean Gaussian noise with variance sqrt(0.1)
# i.e. if the distribution is independant then the covariance matrix is diagonal
# lastly, apply the noisy fading chanel to the training data

# partition_size = 10
# frame_size = 500
for partition in range(0, partition_size):
    # 5000 samples are divided into 10 subset. i.e 1 subset has 500 indexes
    index = np.arange((partition * frame_size), (partition + 1) * frame_size)

    # at a given index of a frame, the receiver only has access to a noisy estimate of the fading channel
    # fading channel entries are corrupted by i.i.d. zero-mean Gaussian noise with variance sqrt(0.1)
    # apply the noisy fading chanel estimates to the training data
    train_noise[0, index] = np.fliplr(
        fading_channel + np.sqrt(error_var_est) * np.dot(np.array([np.random.randn(np.size(fading_channel))]),
                                                         np.diag(fading_channel[0, :]))).dot(
        train_bpsk_reshaped[:, index])

v_fSigWdB = np.arange(-6, 10 + 1, 2)

# Generate neural network
learning_rate = 0.00005
num_input = 1
num_hidden_unit = 100
hidden_size = 1

# Original
v_net = viterbinet(input_size=num_input, hidden_unit_size=num_hidden_unit, num_layers=hidden_size,
                   class_size=num_states)

# 2StackLSTM
v_net3 = multiLSTM(input_size=num_input, hidden_unit_size=num_hidden_unit, num_layers=hidden_size,
                   class_size=num_states)
# 2StackGRU
v_net4 = multiGRU(input_size=num_input, hidden_unit_size=num_hidden_unit, num_layers=hidden_size,
                  class_size=num_states)

# CNN
v_net_cnn = CnnViterbiNet()

# Attention
v_net5 = MultiHeadAttention()
#AttentionLSTM()
#MultiHeadAttention()


# start simulation training and testing with different ranges of SNR
for i in range(0, len(v_fSigWdB)):

    # convert snr decibels to normal scale
    sigma_W = pow(10, (-0.1 * v_fSigWdB[i]))  # same as 10log(10) = v_fSigWdB[i]
    train_channel_out = train_bpsk_product + np.sqrt(sigma_W) * np.random.randn(np.size(train_bpsk_product))
    train_noise_channel_out = train_noise + np.sqrt(sigma_W) * np.random.randn(np.size(train_bpsk_product))
    test_channel_out = test_bpsk_product + np.sqrt(sigma_W) * np.random.randn(np.size(test_bpsk_product))

    # ViterbiNet
    if model_curves[0] == 1:
        train_net = v_net.TrainViterbiNet(channel_train=train_channel_out, symbol_train=train_symbol_reshape,
                                              const_size=const_size)

        v_fXhat1 = v_net.TestViterbiNet(channel_train=train_channel_out, train_size=train_size,
                                            channel_test=test_channel_out, test_size=test_size,
                                            const_size=const_size, memory_length=channel_len)


        m_fSER[0, i] = np.mean(v_fXhat1[0, :] != test_symbol)
        print(m_fSER[0, i])
        print('Viterbinet Done')


    # ViterbiNet Noisy
    if model_curves[1] == 1:
        train_net2 = v_net.TrainViterbiNet(channel_train=train_noise_channel_out, symbol_train=train_symbol_reshape,
                                               const_size=const_size)

        v_fXhat2 = v_net.TestViterbiNet(channel_train=train_noise_channel_out, train_size=train_size,
                                            channel_test=test_channel_out, test_size=test_size, const_size=const_size,
                                            memory_length=channel_len)

        m_fSER[1, i] = np.mean(v_fXhat2[0, :] != test_symbol)
        print(f"ViterbiNet Noise Done at SNR {v_fSigWdB[i]} = {m_fSER[1, i]}")

    # Viterbi
    if model_curves[2] == 1:
        # get the conditional pdf for each state to start viterbi decoder
        state_likelihood = np.array(np.zeros((test_size, num_states)))
        for state in range(num_states):
            # 4x1 vector containing remainder values of each state by the total constellation point
            channel_vector = np.zeros((channel_len, 1))
            state_index = state
            for channel in range(channel_len):
                channel_vector[channel] = state_index % const_size + 1
                # update the index
                state_index = np.floor(state_index / const_size)
            # ensure that the viterbi reciever knows the constellation points (-1, 1)
            v_fS = 2 * (channel_vector - 0.5 * (const_size + 1))
            # conditional pdf of X_test without the constellation point fadding channel
            state_likelihood[:, state] = stats.norm.pdf(test_channel_out - np.fliplr(fading_channel).dot(v_fS), 0,
                                                        sigma_W)
        v_fXhat2 = v_fViterbi(state_likelihood, const_size, channel_len)
        m_fSER[2, i] = np.mean(v_fXhat2[0, :] != test_symbol)
        print(f"Viterbi Done at SNR {v_fSigWdB[i]} = {m_fSER[2, i]}")

    # Viterbi noisy
    if model_curves[3] == 1:

        # print the fading channel (in reverse) 50000 times, adding each row with a gaussian sample with variance
        # sqrt(0.1)
        noise_fChannel = np.array([np.exp(-(np.array([0.001])) * np.arange(0, channel_len, 1))])
        noisy_fChannel = np.tile(np.fliplr(noise_fChannel), [test_size, 1]) + np.sqrt(error_var_est) * np.random.randn(
            test_size, channel_len)
        # get the conditional pdf for each state to start viterbi decoder
        state_likelihood = np.array(np.zeros((test_size, num_states)))
        for state in range(num_states):
            # 4x1 vector containing remainder values of each state by the total constellation point
            channel_vector = np.zeros((channel_len, 1))
            state_index = state
            for channel in range(channel_len):
                channel_vector[channel] = state_index % const_size + 1
                # update the index
                state_index = np.floor(state_index / const_size)
            # ensure that the viterbi reciever knows the constellation points (-1, 1)
            v_fS = 2 * (channel_vector - 0.5 * (const_size + 1))
            # multiply the noisy estimates with a vector correct constellation key (-1, 1) (vector shape: 5x1)
            x = np.fliplr(noisy_fChannel).dot(v_fS)
            # reshape vector to 1x50000
            x.shape = (test_channel_out.shape[0], test_channel_out.shape[1])
            # conditional pdf of test_channel_out without the constellation point fadding channel
            state_likelihood[:, state] = stats.norm.pdf(test_channel_out - x, 0,
                                                        sigma_W)
        v_fXhat4 = v_fViterbi(state_likelihood, const_size, channel_len)
        m_fSER[3, i] = np.mean(v_fXhat4[0, :] != test_symbol)
        print(f"Viterbi Noise Done at SNR {v_fSigWdB[i]} = {m_fSER[3, i]}")
# plot the results
plt.figure()
plt.semilogy(v_fSigWdB, m_fSER[0, :], 'ro--',  # red
             v_fSigWdB, m_fSER[1, :], 'go--',  # green
             v_fSigWdB, m_fSER[2, :], 'bo--',  # blue
             v_fSigWdB, m_fSER[3, :], 'ko--',  # black
             )

plt.legend(
    ('ViterbiNet - Perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi - Perfect CSI', 'Viterbi - CSI uncertainty'))
plt.xlabel('SNR [dB]')
plt.xlim(-6, 10)
plt.ylabel('SER')
plt.grid(True, which="both", ls="-")
plt.show()
print('')
