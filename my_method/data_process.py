import numpy as np
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


def load_ngram_feature_label(data_arr, user2idx, ngram2idx, max_ngram_len):
    text_list, y = load_labeled_data(data_arr, user2idx)
    sample_num = len(text_list)
    x = np.zeros((sample_num, max_ngram_len), dtype=np.uint32)
    for idx, text in enumerate(text_list):
        text_ngram_id = datahelper.text2ngramid(text, ngram2idx, padding=True, max_len=max_ngram_len)
        x[idx, :] = text_ngram_id
    return x, np.array(y)


if __name__ == '__main__':
    pass
    # files = fu.listchildren('/home/yangl/research/authorship/data/twitter/process')
    # for file in files:
    #     twarr = fu.load_array(file)
    #     for tw in twarr:
    #         print(tw['text'])
    #         print(tw['entities']['media'][0]['media_url'])





