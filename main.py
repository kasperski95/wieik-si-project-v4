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
from utils.Word2Vec import Word2Vec

wandb.init(project="wieik-si-project-v4")


MAX_WORDS = 20
WORD_VEC_DIMENSIONS = 300


def load_data():
    data = {"text": [], "is_positive": []}

    for line in Preprocessor("data/neg.txt").run():
        data["text"].append(line)
        data["is_positive"].append(0)

    for line in Preprocessor("data/pos.txt").run():
        data["text"].append(line)
        data["is_positive"].append(1)

    data["text"], data["is_positive"] = shuffle(np.array(data["text"]), np.array(data["is_positive"]))
    data["vectors"] = Word2Vec("data/GoogleNews-vectors-negative300.bin", 10000, MAX_WORDS, WORD_VEC_DIMENSIONS).convert(data["text"])
    data["vectors"] = np.array(data["vectors"])

    return data


def create_model_from_hyperparameters(hp):
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

data = load_data()
X_train, X_test, y_train, y_test = train_test_split(data["vectors"], data["is_positive"], test_size=0.1, random_state=42)

# -------------------------------------------------
# find hyperparameters

# tuner = RandomSearch(create_model_from_hyperparameters, objective="val_accuracy", max_trials=20, directory="models", seed=42)
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
