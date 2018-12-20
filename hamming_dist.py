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
    code_len = A.shape[1]
    A = (2 * A) - 1
    B = (2  * B) - 1
    dists = 0.5 * (code_len - A.dot(B.T))
    return dists.astype("int")

if __name__ == "__main__":
    A = np.array([[0,1,0],[1,1,0]])
    B = np.array([[1,0,1],[1,1,1]])
    # [0,1,0] ^ [1,0,1] => 3
    # [0,1,0] ^ [1,1,1] => 2
    # ...
    output = np.array([[3, 2],[2,1]])
    num_correct = (hamming_dist(A, B) == output).sum()
    assert num_correct == output.size, "Invalid output!"
