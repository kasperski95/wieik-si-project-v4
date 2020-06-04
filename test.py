import argparse
import numpy as np
from tensorflow.keras.models import load_model
from utils.Preprocessor import Preprocessor
from utils.Word2Vec import Word2Vec

MAX_WORDS = 20
WORD_VEC_DIMENSIONS = 300

parser = argparse.ArgumentParser()
parser.add_argument("--review", metavar="This movie sucks!", type=str, required=True)
parser.add_argument("--word2vec-limit", metavar="10000", type=int, required=False)


def main(args):
    data = {}
    data["vectors"] = Word2Vec("data/GoogleNews-vectors-negative300.bin", args.word2vec_limit, MAX_WORDS, WORD_VEC_DIMENSIONS).convert(
        [Preprocessor().preprocess_line(args.review, True)]
    )
    data["vectors"] = np.array(data["vectors"])
    model = load_model("model.h5")

    result = model.predict(data["vectors"])
    print(f"Review is positive: {result[0][0]*100:.2f}%")
    print(f"Review is negative: {100 - result[0][0]*100:.2f}%")


main(parser.parse_args())
