import numpy as np
from scipy import sparse
import utils.key_utils as ku
import utils.data_utils as du


datahelper = du.DataHelper()

def load_labeled_data(data_arr, u2i):
    y = []
    x = []
    Id = ku.reviewer_ID
    text = ku.review_text
    for item in data_arr:
        user = u2i[item[Id]]
        x.append(item[text])
        y.append(user)
    return x, y


def load_n_gram_feature_label(data_arr, ngram2idx, user2idx):
    text_list, y = load_labeled_data(data_arr, user2idx)
    sample_num = len(text_list)
    clo = len(ngram2idx)
    x = np.zeros((sample_num, clo + 1), dtype=bool)
    for idx, text in enumerate(text_list):
        text_ngram_id = datahelper.text2ngramid(text, ngram2idx)
        for i in range(len(text_ngram_id)):
            x[idx, int(text_ngram_id[i])] = True
    x = sparse.csr_matrix(x)
    return x, np.array(y)