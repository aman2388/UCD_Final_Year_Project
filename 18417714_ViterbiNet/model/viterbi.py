import numpy as np


def v_fViterbi(predicted_proba, const_size, memory_length):
    """
    Apply Viterbi detection from computed priors

    Syntax
    -------------------------------------------------------
    viterbi_output = v_fViterbi(m_fPriors, s_nConst, s_nMemSize)

    INPUT:
    -------------------------------------------------------
    predicted_proba - evaluated likelihoods for each state at each time instance
    const_size - constellation size
    memory_length - channel memory length


    OUTPUT:
    -------------------------------------------------------
    viterbi_output - recovered symbols vector
    """

    data_size = len(predicted_proba)
    num_states = pow(const_size, memory_length)

    # Generate trellis matrix (16x2)
    trellis_matrix = np.zeros((num_states, const_size))
    for state in range(0, num_states):
        index = np.mod(state, pow(const_size, (memory_length - 1)))
        for trans in range(0, const_size):
            trellis_matrix[state, trans] = const_size * index + trans

    #### Resulting trellis matrix
    # [0][0] = 0.0
    # [0][1] = 1.0
    # [1][0] = 2.0
    # [1][1] = 3.0
    #     ...
    # [14][0] = 12.0
    # [14][1] = 13.0
    # [15][0] = 14.0
    # [15][1] = 15.0

    # Start Viterbi

    ## matrix of 50000x16 containing likelihoods of each state
    log_priors = -1 * np.log(predicted_proba)

    states = np.zeros((num_states, 1))
    viterbi_output = np.zeros((1, data_size))

    for data_index in range(0, data_size):
        next_state = np.zeros((num_states, 1))
        for state in range(0, num_states):
            # temp 2x1 matrix to compare values from two states of a trellis matrix i.e [0][0] vs [0][1]
            temp = np.zeros((const_size, 1))
            for trans in range(0, const_size):
                # add the values from the trellis matrix and values from the likelihood matrix
                temp[trans] = states[(int(trellis_matrix[state, trans]))] + log_priors[data_index, state]
            # store the min value of the 2x1 temp matrix into the next_state matrix
            next_state[state] = np.min(temp)
        states = next_state
        # find the index that has the min value
        I = np.argmin(states)
        # return index of first symbol in current state
        viterbi_output[0, data_index] = np.mod(I, const_size) + 1

    return viterbi_output
