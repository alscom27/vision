import numpy as np
from tensorflow.keras.utils import to_categorical

actions = ["zero", "one", "two", "three", "four", "five"]

# numpy 배열 합치기
data = np.concatenate(
    [
        np.load("dataset/seq_zero.npy"),
        np.load("dataset/seq_one.npy"),
        np.load("dataset/seq_two.npy"),
        np.load("dataset/seq_three.npy"),
        np.load("dataset/seq_four.npy"),
        np.load("dataset/seq_five.npy"),
    ],
    axis=0,
)

data.shape

x_data = data[:, :, :-1]
labels = data[:, 0, _1]

print(x_data.shape)
print(labels.shape)

y_data = to_categorical(labels, num_classes=len(actions))

y_data.shape

from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.1, random_state=2021
)

# xtrain 등등 출력해보기

####

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential(
    [
        LSTM(64, activation="relu", input_shape=x_train.shape[1:3]),
        Dense(32, activation="relu"),
        Dense(len(actions), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint(
            "models/model.h5",
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
            mode="auto",
        ),
        ReduceLROnPlateau(
            monitor="val_acc", factor=0.5, patience=50, verbose=1, mode="auto"
        ),
    ],
)

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history["loss"], "y", label="train loss")
loss_ax.plot(history.history["val_loss"], "r", label="val loss")
loss_ax.set_xlabel("epoch")
loss_ax.set_ylabel("loss")
loss_ax.legend(loc="upper left")

acc_ax.plot(history.history["acc"], "b", label="train acc")
acc_ax.plot(history.history["val_acc"], "g", label="val acc")
acc_ax.set_ylabel("accuracy")
acc_ax.legend(loc="upper left")

plt.show()
