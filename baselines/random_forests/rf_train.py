from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
import utils.data_utils as du
import sklearn.utils as sku
import baselines.random_forests.data_process as dp
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


datahelper = du.DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = du.UserHelper()


reviews  = du.ReviewLoader(ku.Movie, product_num=100).get_data()
users = userhelper.get_users(reviews)


user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=2)
# voca.dump_n_grams(ngram2idx, type=ku.charngram2idx)

grid_params= {'n_estimators': [1000], 'criterion': ['gini'], 'max_depth': [100],
            'min_samples_split': [3],
         'min_samples_leaf': [1], 'max_features': ['sqrt'], 'max_leaf_nodes': [None],
         'min_impurity_decrease': [0.], 'random_state': [1]}
rf_params = {'n_jobs': 10, 'verbose': 0, 'warm_start': False}

rfc = RFC(**rf_params)


gs_param = {'scoring': 'accuracy', 'param_grid': grid_params, 'estimator': rfc, 'cv':10, }



epoch = 1
#
def test(classifier, text_x, test_y):
    y_pred = classifier.predict(text_x)
    acc = accuracy_score(y_true=test_y, y_pred=y_pred)
    return acc
#
#
#
reviews = sku.shuffle(reviews)
training_split = int(len(reviews) * 0.8)
valid_split = training_split - int(training_split * 0.2)


for i in range(epoch):
    reviews = sku.shuffle(reviews)
    training_data = reviews[:valid_split]
    valid_data = reviews[valid_split:training_split]
    training_data += valid_data
    testing_data = reviews[training_split:]
    print('epoch {}'.format(i + 1))


    training_x, training_y = dp.load_n_gram_feature_label(training_data, ngram2idx, user2idx)
    valid_x, valid_y = dp.load_n_gram_feature_label(valid_data, ngram2idx, user2idx,)
    testing_x, testing_y = dp.load_n_gram_feature_label(testing_data, ngram2idx, user2idx)
    print(training_x.shape, training_y.shape)
    clf = GridSearchCV(**gs_param)
    clf.fit(training_x, training_y)
    print(clf.best_params_)

    acc = test(clf, testing_x, testing_y)
    print(acc)


