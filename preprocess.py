
import gensim
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.engine.sequential import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # lemmatization

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
lemmatizer = nltk.wordnet.WordNetLemmatizer()

MAX_WORDS = 20
WORD_VEC_DIMENSIONS = 300

def preprocess(filepath):
    result = []

    with open(filepath) as file:
        for line in file:
            tokens = tokenizer.tokenize(line)
            tokens_reduced = [w for w in tokens if not w in stop_words]
            # tokens_reduced_stemmed = [stemmer.stem(w) for w in tokens_reduced ]
            tokens_reduced_lemmatized = [lemmatizer.lemmatize(w) for w in tokens_reduced ]
            result.append(tokens_reduced_lemmatized)

    return result

def convert_words_to_vectors(word2vec, words):
    results = [[0 for col in range(MAX_WORDS)] for row in range(WORD_VEC_DIMENSIONS)]
    i = 0
    for word in words:
        if i >= MAX_WORDS: break

        if not word in word2vec:
            continue
        else:
            results[i] = word2vec[word]
            i += 1
    return results


data = {"text": [], "is_positive": []}

for line in preprocess("data/pos.txt"):
    data["text"].append(line)
    data["is_positive"].append(0)

for line in preprocess("data/neg.txt"):
    data["text"].append(line)
    data["is_positive"].append(1)

data["text"], data["is_positive"] = shuffle(np.array(data["text"]), np.array(data["is_positive"]))


word2vec = gensim.models.KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True, limit=10000)
data["vectors"] = []
for line in data["text"]:
    data["vectors"].append(convert_words_to_vectors(word2vec, line))

df = pd.DataFrame(data)

# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(data["vectors"], data["is_positive"], test_size=0.33, random_state=42)

print(pd.DataFrame(X_train))
print(np.array(X_train).shape)


# model = Sequential()
# model.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(5, activation = 'softmax'))
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='Adam',
#     metrics=['accuracy']
# )