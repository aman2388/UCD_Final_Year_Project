import numpy as np

class matrix_reshape():

    def __init__(self):
        pass

    def reshape_data(self, vector, memory):
        num_cols = np.size(vector)
        matrix = np.ones((memory, num_cols))

        # reshape rows from bottom to top
        for row in range(0, memory):
            index = memory - row - 1
            matrix[index, 0:num_cols - index-1] = vector[0, index: num_cols-1]

        return matrix

    def shift_elements(self, arr: np.ndarray, memory):
        arr_cpy = np.array(arr)
        # assign the same vector 4 times to represent matrix
        matrix = np.vstack([arr_cpy] * memory)
        for i in range(matrix.shape[0]):
            matrix[i, :] = np.roll(matrix[i, :], -i)
        return matrix




