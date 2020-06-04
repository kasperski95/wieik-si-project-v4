import argparse
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
from utils.Preprocessor import Preprocessor
from utils.SaveCallback import SaveCallback
from utils.Word2Vec import Word2Vec

wandb.init(project="wieik-si-project-v4")

MAX_WORDS = 20
WORD_VEC_DIMENSIONS = 300

parser = argparse.ArgumentParser()
parser.add_argument("--word2vec-limit", metavar="10000", type=int, required=False)
parser.add_argument("--search-hyperparameters", metavar="False", type=bool, required=False)


def load_data(word2vec_limit):
    data = {"text": [], "is_positive": []}

    preprocessor = Preprocessor()

    for line in preprocessor.preprocess_file("data/neg.txt"):
        data["text"].append(line)
        data["is_positive"].append(0)

    for line in preprocessor.preprocess_file("data/pos.txt"):
        data["text"].append(line)
        data["is_positive"].append(1)

    preprocessor.visualize()

    data["text"], data["is_positive"] = shuffle(np.array(data["text"]), np.array(data["is_positive"]))
    data["vectors"] = Word2Vec("data/GoogleNews-vectors-negative300.bin", word2vec_limit, MAX_WORDS, WORD_VEC_DIMENSIONS).convert(
        data["text"]
    )
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


def find_hyperparameters(X_train, X_test, y_train, y_test):
    tuner = RandomSearch(create_model_from_hyperparameters, objective="val_accuracy", max_trials=20, directory="models", seed=42)
    tuner.search(x=X_train, y=y_train, epochs=25, validation_data=(X_test, y_test))
    tuner.results_summary()


def create_final_model():
    model = Sequential()
    model.add(LSTM(MAX_WORDS, input_shape=(MAX_WORDS, WORD_VEC_DIMENSIONS), activation="tanh", dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    return model


# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


def main(args):
    data = load_data(args.word2vec_limit)
    X_train, X_test, y_train, y_test = train_test_split(data["vectors"], data["is_positive"], test_size=0.1, random_state=42)

    if args.search_hyperparameters:
        find_hyperparameters(X_train, X_test, y_train, y_test)
    else:
        model = create_final_model()
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


main(parser.parse_args())
