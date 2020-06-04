import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer


class Preprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")  # lemmatization

        self._tokenizer = RegexpTokenizer(r"\w+")
        self._stop_words = set(stopwords.words("english"))
        # self._stemmer = nltk.stem.SnowballStemmer("english")
        self._lemmatizer = nltk.wordnet.WordNetLemmatizer()

    def preprocess_file(self, filepath):
        results = []

        with open(filepath) as file:
            for line in file:
                results.append(self.preprocess_line(line))

        return results

    def preprocess_line(self, line):
        tokens = self._tokenizer.tokenize(line)
        tokens_reduced = [w for w in tokens if not w in self._stop_words]
        # tokens_reduced_stemmed = [stemmer.stem(w) for w in tokens_reduced ]
        tokens_reduced_lemmatized = [self._lemmatizer.lemmatize(w) for w in tokens_reduced]
        return tokens_reduced_lemmatized
