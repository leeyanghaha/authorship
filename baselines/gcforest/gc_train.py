from baselines.gcforest.GcForest import GCForest
from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
from utils.data_utils import ReviewLoader, FeatureLoader, DataHelper, UserHelper
import sklearn.utils as sku
from sklearn.metrics import accuracy_score
import os
import pickle


datahelper = DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = UserHelper()
feature_loader = FeatureLoader()

reviews  = ReviewLoader(ku.Movie, product_num=50).get_data()
users = userhelper.get_users(reviews)


user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=2)
voca.dump_n_grams(ngram2idx, type=ku.charngram2idx)


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 203
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

def test(test_x, test_y, gc):
    y_pred = gc.predict(test_x)
    return accuracy_score(test_y, y_pred)


def save(gc):
    path = os.path.join(ku.model_root, 'gcf.pkl')
    if os.path.exists(path):
        os.system('rm {}'.format(path))
    else:
        with open(path, 'wb') as f:
            pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)

def load():
    path = os.path.join(ku.model_root, 'gcf.pkl')
    with open(path, 'rb') as f:
        gc = pickle.load(f)
    return gc


config = get_toy_config()
gc = GCForest(config)


training_split = int(len(reviews) * 0.8)
valid_split = training_split - int(training_split * 0.2)

reviews = sku.shuffle(reviews)
training_data = reviews[:valid_split]
valid_data = reviews[valid_split:training_split]
testing_data = reviews[training_split:]


training_x, training_y = feature_loader.load_n_gram_binary_feature_label(training_data, ngram2idx, user2idx, sparse_tag=False)
valid_x, valid_y = feature_loader.load_n_gram_binary_feature_label(valid_data, ngram2idx, user2idx, sparse_tag=False)
testing_x, testing_y = feature_loader.load_n_gram_binary_feature_label(testing_data, ngram2idx, user2idx, sparse_tag=False)


gc.fit_transform(training_x, training_y, X_test=valid_x, y_test=valid_y)

acc = test(valid_x, valid_y, gc)
save(gc)

print('acc: ', acc)