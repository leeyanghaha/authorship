from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
import utils.key_utils as ku


PSD = ' '
PED = ' '


class Ngram(object):
    def __init__(self, n=4):
        self.n = n

    def character_level(self, text, n=3):
        text = PSD + text + PED
        n = self.n if n is None else n
        character_n_grams = [text[i:i+n] for i in range(len(text) - n + 1)]
        return character_n_grams

    def word_level(self, text, n=2):
        ngram_vector = CountVectorizer(ngram_range=(n, n), decode_error='ignore',
                                       token_pattern=r'\b\w+\b', min_df=1)
        ngram_vector.fit_transform([text])
        return list(ngram_vector.vocabulary_.keys())


def load_n_gram_feature_label(text_list, user_id_list, ngram2idx, user2idx, to_sparse=True):
    '''
    将text转化为n_gram_id 矩阵, 将user 转化成其 user_id
    :param text: list text 数据
    :param user: user id list
    :param to_sparse: 是否将text转化成为稀疏矩阵
    :return:
    '''
    sample_num = len(text_list)
    clo = len(ngram2idx)
    x = np.zeros((sample_num, clo), dtype=bool)
    for idx, text in enumerate(text_list):
        text_ngram_id = text2character_n_gram(text, ngram2idx)
        for i in range(len(text_ngram_id)):
            x[idx, int(text_ngram_id[i])] = True
    if to_sparse:
        x = sparse.csr_matrix(x)
    return x, np.array(user_id_list)


def text2character_n_gram(text, ngram2idx):
    ngram = Ngram()
    ngram_list = ngram.character_level(text)
    text_id = []
    for gram in ngram_list:
        if gram in ngram2idx:
            text_id.append(ngram2idx[gram])
        else:
            text_id.append(ngram2idx[ku.UNK])
    return text_id



if __name__ == '__main__':
    ngram = Ngram()
    print(ngram.word_level('tedfsasda fdsafhuihfd fdsaif fdasf fdasfa fdasf fdsafa f'))