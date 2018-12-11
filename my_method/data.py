import utils.function_utils as fu
import utils.key_utils as ku
import utils.data_utils as du
import numpy as np
import utils.photo_utils as phu
import os
import sklearn.utils as sku


class TweetImageDataLoader:
    def __init__(self, data_arr, user2idx, ngram2idx, max_ngram_len):
        self.data_arr = data_arr
        self.user2idx = user2idx
        self.ngram2idx = ngram2idx
        self.max_ngram_len = max_ngram_len
        self.dataheler = du.DataHelper(ku.twitter)

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        text_ngram_id = self.dataheler.text2ngramid(item[ku.twitter_text], self.ngram2idx, padding=True,
                                                    max_len=self.max_ngram_len)
        user_id = self.user2idx[item[ku.twitter_user]]
        image = item[ku.image]
        sample = {ku.text_id: text_ngram_id, ku.user_id: user_id, ku.image: image}
        return sample

def load_feature_label(data_arr, ngram2idx, user2idx, max_ngram_len):
    loader = TweetImageDataLoader(data_arr, user2idx, ngram2idx, max_ngram_len)
    text = []
    image = []
    user = []
    for item in loader:
        text.append(item[ku.text_id])
        image.append(item[ku.image])
        user.append(item[ku.user_id])
    return np.array(text), np.array(image), np.array(user)


def load_tweets(tweets_image):
    res = []
    for tweet_image in tweets_image:
        text = tweet_image['text']
        res.append({ku.twitter_text: text})
    return res


def get_media_users(user_num):
    users = fu.listchildren(ku.photo_dir, concat=False)
    return users[:user_num]


if __name__ == '__main__':
    users = get_media_users(50)




