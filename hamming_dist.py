import numpy as np
from pdb import set_trace

def hamming_dist(A, B):
    '''
    Calculates the hamming distance between two arrays with binary values.

    Args:
        - A (numpy.ndarray): array of shape (N, C) where N is the number of
            samples and C is the number of binary codes.
        - B (numpy.ndarray): array of shape (M, C) where M is the number of
            samples and C is the number of binary codes.

    Returns:
        (numpy.ndarray): array of shape (N, M), where the ith N and jth M is
            the hamming distance between the ith A and jth B.
    '''
    num_A, num_B = A.shape[0], B.shape[0]
    result = np.zeros((num_A, num_B), dtype=np.int16)

    for i in range(num_A):
        diff = np.bitwise_xor(A[i], B)
        dist = diff.sum(axis=1)
        result[i] = dist

    return result

if __name__ == "__main__":
    A = np.array([[0,1,0],[1,1,0]])
    B = np.array([[1,0,1],[1,1,1]])
    # [0,1,0] ^ [1,0,1] => 3
    # [0,1,0] ^ [1,1,1] => 2
    # ...
    output = np.array([[3, 2],[2,1]])
    num_correct = (hamming_dist(A, B) == output).sum()
    assert num_correct == output.size, "Invalid output!"
