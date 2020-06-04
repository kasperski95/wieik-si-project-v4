import os

import gensim
import numpy as np
import pandas as pd
import wandb
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from wandb.keras import WandbCallback

from utils.SaveCallback import SaveCallback
from utils.Preprocessor import Preprocessor

wandb.init(project="wieik-si-project-v4")


MAX_WORDS = 20
WORD_VEC_DIMENSIONS = 300


def convert_words_to_vectors(word2vec, words):
    results = [[0 for meaning in range(WORD_VEC_DIMENSIONS)] for word in range(MAX_WORDS)]
    i = 0
    for word in words:
        if i >= MAX_WORDS:
            break

        if not word in word2vec:
            continue
        else:
            results[i] = word2vec[word]
            i += 1
    return results


def create_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            MAX_WORDS,
            input_shape=(MAX_WORDS, WORD_VEC_DIMENSIONS),
            activation=hp.Choice("activation", ["relu", "tanh"]),
            dropout=hp.Float("dropout", 0.0, 0.4, 0.1),
            recurrent_dropout=hp.Float("recurrent_dropout", 0.0, 0.4, 0.1),
        )
    )
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=hp.Choice("optimizer", ["RMSProp", "Adam", "Ftrl"]), metrics=["accuracy"],
    )

    return model


# -------------------------------------------------
# preprocessing

data = {"text": [], "is_positive": []}

for line in Preprocessor("data/pos.txt").run():
    data["text"].append(line)
    data["is_positive"].append(0)

for line in Preprocessor("data/pos.txt").run():
    data["text"].append(line)
    data["is_positive"].append(1)

data["text"], data["is_positive"] = shuffle(np.array(data["text"]), np.array(data["is_positive"]))


word2vec = gensim.models.KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True, limit=10000)
data["vectors"] = []
for line in data["text"]:
    data["vectors"].append(convert_words_to_vectors(word2vec, line))

# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(data["vectors"], data["is_positive"], test_size=0.1, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)

# -------------------------------------------------
# find hyperparameters

# tuner = RandomSearch(create_model, objective="val_accuracy", max_trials=20, directory="models", seed=42)
# tuner.search(x=X_train, y=y_train, epochs=25, validation_data=(X_test, y_test))
# tuner.results_summary()

# -------------------------------------------------
# train

model = Sequential()
model.add(LSTM(MAX_WORDS, input_shape=(MAX_WORDS, WORD_VEC_DIMENSIONS), activation="tanh", dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics="accuracy",
)


model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[SaveCallback("model", 25), WandbCallback()],
)
model.save("model.h5")
