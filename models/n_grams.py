from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
import utils.key_utils as ku


PSD = ' '
PED = ' '


class Ngram(object):
    def __init__(self, n=3):
        self.n = n

    def character_level(self, text, n=None):
        text = PSD + text + PED
        n = self.n if n is None else n
        character_n_grams = [text[i:i+n] for i in range(len(text) - n + 1)]
        return character_n_grams

    def word_level(self, text, n=2):
        ngram_vector = CountVectorizer(ngram_range=(n, n), decode_error='ignore',
                                       token_pattern=r'\b\w+\b', min_df=1)
        ngram_vector.fit_transform([text])
        return list(ngram_vector.vocabulary_.keys())




if __name__ == '__main__':
    ngram = Ngram()
    print(ngram.word_level('tedfsasda fdsafhuihfd fdsaif fdasf fdasfa fdasf fdsafa f'))