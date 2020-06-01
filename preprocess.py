
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.utils import shuffle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # lemmatization

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
lemmatizer = nltk.wordnet.WordNetLemmatizer()

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



    
data = {"text": [], "is_positive": []}

for line in preprocess("data/pos.txt"):
    data["text"].append(line)
    data["is_positive"].append(True)

for line in preprocess("data/neg.txt"):
    data["text"].append(line)
    data["is_positive"].append(False)



data["text"], data["is_positive"] = shuffle(np.array(data["text"]), np.array(data["is_positive"]))
print(data)
