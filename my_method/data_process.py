import utils.function_utils as fu
import utils.key_utils as ku
import utils.pattern_utils as pu
import os


def pre_process(twarr):
    # 去除 retweet, 去除
    res = []

    for tw in twarr:
        if tw[ku.retweet] == False:
            text = pu.text_normalization(tw[ku.twitter_text])
            tw[ku.twitter_text] = text
            res.append(tw)
    return res



if __name__ == '__main__':
    pass
    # files = fu.listchildren('/home/yangl/research/authorship/data/twitter/process')
    # for file in files:
    #     twarr = fu.load_array(file)
    #     for tw in twarr:
    #         print(tw['text'])
    #         print(tw['entities']['media'][0]['media_url'])





