import utils.function_utils as fu
from my_method.my_capsule.input import ReviewDataSet, ReviewInfo

def get_reviews():
    file = r'/home/leeyang/research/data/Movie.json'
    reviews = fu.load_array(file)
    return reviews


reviews = get_reviews()
# reviews = ReviewLoader(ku.Movie, product_num=50).get_data()
review_info = ReviewInfo(reviews, max_len=500, min_threshold=0, feature_name='word')

num_classes = review_info.num_classes
vocab_size = review_info.vocab_size

feature2idx = review_info.feature2idx
user2idx = review_info.user2idx


def get_idx2feature(feature2idx):
    res = dict()
    for ngram, idx in feature2idx.items():
        res.update({idx: ngram})
    return res


idx2feature = get_idx2feature(feature2idx)
idx2user = get_idx2feature(user2idx)


x_ids, y_ids = review_info.x, review_info.y

# print(x_ids.shape)

def recover_text(idx2feature):
    texts = []
    for i in range(x_ids.shape[0]):
        ids = x_ids[i, :]
        text = ''
        for j, id in enumerate(ids):
            if j == 0:
                text += idx2feature[id]
            else:
                text += idx2feature[id][-1]
        texts.append(text)
    return texts


def recover_user(idx2user):
    user = []
    for i in y_ids:
        user.append(idx2user[i])
    return user


recover_x = recover_text(idx2feature)
recover_y = recover_user(idx2user)

for x, y in zip(recover_x, recover_y):
    print(x, y)
