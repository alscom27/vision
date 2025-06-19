import numpy as np
import scipy.special as sp


def self_attention(X, Wq, Wk, Wv, d_key):
    Q = np.matmul(X, Wq)
    K = np.matmul(X, Wk)
    V = np.matmul(X, Wv)

    QK = np.matmul(Q, np.transpose(K))
    A = sp.softmax(QK / np.sqrt(d_key), axis=1)
    C = np.matmul(a, V)

    return Q, K, V, A, C


X = np.asarray(
    [
        [0.0, 0.6, 0.3, 0],
        [0.1, 0.9, 0.0, 0.0],
        [0, 0.1, 0.8, 0.1],
        [0.3, 0.0, 0.6, 0.0],
        [0.0, 0.1, 0.0, 0.9],
    ]
)

Wq = np.asarray([[1, 0], [1, 0], [0, 1], [0, 3]])
Wk = np.asarray([[0, 1], [1, 0], [1, 0], [0, 2]])
Wv = np.asarray([[1, 2], [0, 1], [1, 0], [0, 0]])

Q, K, V, A, C = self_attention(X, Wq, Wk, Wv, 2)
