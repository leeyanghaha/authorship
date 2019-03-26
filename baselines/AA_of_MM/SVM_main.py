from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
import utils.data_utils as du
import sklearn.utils as sku
from sklearn.model_selection import train_test_split, cross_validate
import baselines.AA_of_MM.data_process as dp
from models.svm_classifier import Svm
import utils.function_utils as fu
from baselines.AA_of_MM.input import ReviewInfo
from job_prediction.job_input import TwitterInfo
datahelper = du.DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = du.UserHelper()


def get_reviews():
    # file = '/home/leeyang/research/data/Movie.json'
    file = '/home/leeyang/research/data/twitter/training_bert'
    return sku.shuffle(fu.load_array(file))

reviews = get_reviews()
# reviews = du.ReviewLoader(ku.Movie, product_num=100).get_data()
# inputs = ReviewInfo(reviews)
inputs = TwitterInfo(reviews, 'n-gram', max_seq_len=3500)
parameters = {'penalty': 'l2', 'loss': 'hinge', 'dual': True, 'multi_class': 'ovr',
                  'fit_intercept': True, 'max_iter': 10000}


epoch = 10


def test(classifier, text_x, test_y):
    y_pred = classifier.predict(text_x)
    acc = classifier.accuracy(test_y, y_pred)
    # print('test acc: %.3f' % acc)
    # print('y_true: ', test_y[:100])
    # print('y_pred: ', y_pred[:100])
    return acc


classifier = Svm(**parameters)


def cv(classifier, x, y):
    total = 0
    for i in range(5):
        x_train, x_test, y_train, y_test = cross_validate(classifier, x, y, cv=10, scoring='accuracy')
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc = classifier.accuracy(y_test, y_pred)
        print('%d: %.3f' % (i, acc))
        total += acc
    print('test acc: %.3f' % (total / 5))


print('test score: ', classifier.cv(inputs.x, inputs.jobs))











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