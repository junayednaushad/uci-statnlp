import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    viterbi = np.ones((N,L)) * -np.inf
    viterbi[0, : ] = start_scores + emission_scores[0, : ]
    back_pointer = np.zeros((N,L))

    for i in range(1,N):
        for j in range(L):
            for k in range(L):
                score = viterbi[i-1, k] + emission_scores[i, j] + trans_scores[k, j]
                if score > viterbi[i, j]:
                    viterbi[i, j] = score
                    back_pointer[i, j] = k

    viterbi[N-1, : ] += end_scores
    y = []
    y.append(np.argmax(viterbi[N-1, : ]))
    s = viterbi[N-1, int(y[0])]

    for i in range(1, N):
        prev_lbl = int(y[-1])
        y.append(back_pointer[N-i, prev_lbl])
    
    y.reverse()

    return (s, y)
