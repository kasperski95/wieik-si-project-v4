import gensim


class Word2Vec:
    def __init__(self, path, limit, max_words, word2vec_dimensions):
        self._word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
        self._max_words = max_words
        self._word2vec_dimensions = word2vec_dimensions

    def convert(self, data):
        results = []
        for line in data:
            results.append(self.convert_words_to_vectors(line))
        return results

    def convert_words_to_vectors(self, words):
        results = [[0 for meaning in range(self._word2vec_dimensions)] for word in range(self._max_words)]
        i = 0
        for word in words:
            if i >= self._max_words:
                break

            if not word in self._word2vec:
                continue
            else:
                results[i] = self._word2vec[word]
                i += 1
        return results
