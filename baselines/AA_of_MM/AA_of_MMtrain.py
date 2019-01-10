from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
import utils.data_utils as du
# import sklearn.utils as sku
# import baselines.AA_of_MM.data_process as dp
# from models.svm_classifier import Svm
import utils.function_utils as fu

datahelper = du.DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = du.UserHelper()


reviews  = du.ReviewLoader(ku.Movie, product_num=100).get_data()
users = userhelper.get_users(reviews)

fu.dump_array(reviews, '/home/yangl/research/authorship/data/review/product/input/all')
# user2idx = userhelper.user2idx(users)
# ngram2idx = voca.character_n_gram_table(reviews, min_threshold=2)
# voca.dump_n_grams(ngram2idx, type=ku.ngram2idx)
#
#
# print('len user: ', len(user2idx))
# print('len ngram: ', len(ngram2idx))
# print('len reviews: ', len(reviews))
#
#
#
# parameters = {'penalty': 'l2', 'loss': 'hinge', 'dual': True, 'multi_class': 'ovr',
#                   'fit_intercept': True, 'max_iter': 1000}
#
#
# epoch = 10
#
# def test(classifier, text_x, test_y):
#     y_pred = classifier.predict(text_x)
#     print('testing......')
#     acc = classifier.accuracy(test_y, y_pred)
#     print('test acc: ', acc)
#     print('y_true: ', test_y[:100])
#     print('y_pred: ', y_pred[:100])
#     return acc
#
#
#
# reviews = sku.shuffle(reviews)
# training_split = int(len(reviews) * 0.8)
# valid_split = training_split - int(training_split * 0.2)
# total_acc = 0
#
# for i in range(epoch):
#     reviews = sku.shuffle(reviews)
#     training_data = reviews[:valid_split]
#     valid_data = reviews[valid_split:training_split]
#     testing_data = reviews[training_split:]
#     print('epoch {}'.format(i + 1))
#
#
#     training_x, training_y = dp.load_n_gram_feature_label(training_data, ngram2idx, user2idx)
#     valid_x, valid_y = dp.load_n_gram_feature_label(valid_data, ngram2idx, user2idx,)
#     testing_x, testing_y = dp.load_n_gram_feature_label(testing_data, ngram2idx, user2idx)
#
#     classifier = Svm(**parameters)
#     classifier.fit(training_x, training_y)
#     y_pred = classifier.predict(valid_x)
#     print('valid......')
#     acc = classifier.accuracy(valid_y, y_pred)
#     print('valid accuracy: {}'.format(acc))
#     testing_acc = test(classifier, testing_x, testing_y)
#     total_acc += testing_acc
#
# print('average acc: ', total_acc / epoch)








# if __name__ == '__main__':
#     x, y = load_data(reviews)
#     training_x, training_y = x[:1000], y[:1000]
#     valid_x, valid_y = x[1000:1200], y[1000:1200]
#     print(valid_y.shape, valid_x.shape)
#     parameters = {'penalty': 'l2', 'loss': 'squared_hinge', 'dual': True, 'multi_class': 'ovr',
#                   'fit_intercept': True, 'max_iter': 1000}
#     classifier = Svm(**parameters)
#     classifier.fit(training_x, training_y)
#     y_pred = classifier.predict(valid_x)
#     print(classifier.accuracy(valid_y, y_pred))