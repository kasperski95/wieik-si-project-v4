import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from wordcloud import WordCloud


class Preprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")  # lemmatization

        self._tokenizer = RegexpTokenizer(r"\w+")
        self._stop_words = set(stopwords.words("english"))
        # self._stemmer = nltk.stem.SnowballStemmer("english")
        self._lemmatizer = nltk.wordnet.WordNetLemmatizer()
        self._vocabulary = set()

    def preprocess_file(self, filepath):
        results = []

        with open(filepath) as file:
            for line in file:
                results.append(self.preprocess_line(line))

        return results

    def preprocess_line(self, line, verbose=False):
        if verbose:
            print(f"RAW: {line}")

        tokens = self._tokenizer.tokenize(line)
        if verbose:
            print(f"TOKENS: {tokens}")

        tokens_reduced = [w for w in tokens if not w in self._stop_words]
        if verbose:
            print(f"TOKENS_REDUCED: {tokens_reduced}")

        # tokens_reduced_stemmed = [stemmer.stem(w) for w in tokens_reduced ]
        tokens_reduced_lemmatized = [self._lemmatizer.lemmatize(w) for w in tokens_reduced]
        if verbose:
            print(f"TOKENS_REDUCED_LEMMATIZED: {tokens_reduced_lemmatized}")

        tokens_reduced_lemmatized_lowercased = [t.lower() for t in tokens_reduced_lemmatized]
        if verbose:
            print(f"TOKENS_REDUCED_LEMMATIZED_LOWERCASED: {tokens_reduced_lemmatized_lowercased}")

        for token in tokens_reduced_lemmatized_lowercased:
            self._vocabulary.add(token)

        return tokens_reduced_lemmatized

    def visualize(self):
        frequency_dist = nltk.FreqDist(self._vocabulary)
        for i in range(1, 6):
            wordcloud = WordCloud(width=1366, height=720, background_color="white").generate_from_frequencies(frequency_dist)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.savefig(f"wordcloud-{i}.png", dpi=300)
