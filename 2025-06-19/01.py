import random
import numpy as np

X = [5, 3, 5, 5, 2, 4, 1, 6, 3, 1]  # 데이터 셋

eyes = [1, 2, 3, 4, 5, 6]
p = np.zeros([6])


def learn_generator(X, p):
    for i in range(len(X)):
        p[X[i] - 1] += 1
    p = p / len(X)


def generate():
    return random.choices(eyes, p)


learn_generator(X, p)
print(generate(), generate(), generate(), generate())
