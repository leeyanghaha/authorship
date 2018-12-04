from baselines.LDAH_S import data
import utils.review_data_utils as du
import utils.key_utils as ku


max_user_num = 11
n_splits = 10


# num_reviews_per_user = data.num_reviews_per_user
# training_num_reviews_per_user = int(num_reviews_per_user * 0.6)
# valid_num_reviews_per_user = int(num_reviews_per_user * 0.2)
# testing_num_reviews_per_user = int(num_reviews_per_user * 0.2)



dataloader = data.dataloader
datahelper = du.DataHelper()
userhelper = du.UserHelper()

reviews = dataloader.load_domain_reviews()
users = datahelper.get_users(reviews)
users = userhelper.sample_user(users, max_user_num)
user2idx = du.DataHelper().user2idx(users)
print(len(user2idx))
reviews_ordered_by_users = data.get_reviews(users)

training_x = data.get_training_feature(reviews_ordered_by_users, user2idx)
print('train_x: ', training_x)
print('train_x: ', training_x.max(axis=1))
print('train_x: ', training_x.argmax(axis=1))
print('training_x over.')
valid_x, valid_y = data.get_valid_feature(reviews_ordered_by_users, user2idx)
print('valid_x: ', valid_x)
print('valid_y: ', valid_y)
print('valid_x over.', valid_x.shape, len(valid_y))
test_x, test_y = data.get_test_feature(reviews_ordered_by_users, user2idx)
print('test_x over.', test_x.shape, len(test_y))


def valid(valid_x, valid_y, training_x):
    assert valid_x.shape[0] == len(valid_y)
    y_pred = data.predict(valid_x, training_x)
    acc = data.accuracy(valid_y, y_pred)
    return  acc


if __name__ == '__main__':
    print(valid(valid_x, valid_y, training_x))
    # pass









