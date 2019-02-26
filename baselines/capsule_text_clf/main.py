from __future__ import division, print_function, unicode_literals
import argparse
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from baselines.capsule_text_clf.loss import spread_loss, cross_entropy, margin_loss
from baselines.capsule_text_clf.network import baseline_model_kimcnn, baseline_model_cnn, capsule_model_A, capsule_model_B
from sklearn.utils import shuffle

from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
from utils.data_utils import UserHelper, ReviewLoader, FeatureLoader
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--embedding_type', type=str, default='rand',
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

# parser.add_argument('--dataset', type=str, default='reuters_multilabel_dataset',
#                     help='Options: reuters_multilabel_dataset, MR_dataset, SST_dataset')

parser.add_argument('--loss_type', type=str, default='cross_entropy',
                    help='margin_loss, spread_loss, cross_entropy')

parser.add_argument('--model_type', type=str, default='CNN',
                    help='CNN, KIMCNN, capsule-A, capsule-B')

parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

parser.add_argument('--use_orphan', type=bool, default='True', help='Add orphan capsule or not')
parser.add_argument('--use_leaky', type=bool, default='False', help='Use leaky-softmax or not')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')#CNN 0.0005
parser.add_argument('--margin', type=float, default=0.2, help='the initial value for spread loss')

import json
args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent=2))


max_len = 3500
feature_name = 'n-gram'


voca = Vocabulary(ku.voca_root)
userhelper = UserHelper()
reviews = ReviewLoader(ku.Movie, product_num=50).get_data()

users = userhelper.get_users(reviews)
user2idx = userhelper.user2idx(users)


def get_feature(reviews):
    if feature_name == 'n-gram':
        feature2idx = voca.character_n_gram_table(reviews, min_threshold=6)
    else:
        feature2idx = voca.word_table(reviews, min_threshold=5)
    feature_loader = FeatureLoader(user2idx=user2idx, max_ngram_len=max_len,
                                   ngram2idx=feature2idx)
    X, Y = feature_loader.load_n_gram_idx_feature_label(reviews)
    return X, Y, len(feature2idx)


def load_data():
    text, label, vocab_size = get_feature(reviews)
    training_split = int(len(reviews) * 0.8)
    valid_split = training_split - int(training_split * 0.2)
    train, train_label =  list(text[: valid_split, :]), list(label[: valid_split])
    dev, dev_label = list(text[valid_split: training_split, :]), list(label[valid_split: training_split])
    test, test_label = list(text[training_split: , :]), list(label[training_split: ])
    w2v = None
    return train, train_label, test, test_label, dev, dev_label, w2v, vocab_size
#
#
class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, dataset, label, batch_size, is_shuffle=True):
      self._dataset = dataset
      self._label = label
      self._batch_size = batch_size
      self._cursor = 0

      if is_shuffle:
          index = np.arange(len(self._dataset))
          np.random.shuffle(index)
          self._dataset = np.array(self._dataset)[index]
          self._label = np.array(self._label)[index]
      else:
          self._dataset = np.array(self._dataset)
          self._label = np.array(self._label)
    def next(self):
      if self._cursor + self._batch_size > len(self._dataset):
          self._cursor = 0
      """Generate a single batch from the current cursor position in the data."""
      batch_x = self._dataset[self._cursor : self._cursor + self._batch_size, :]
      batch_y = self._label[self._cursor : self._cursor + self._batch_size]
      self._cursor += self._batch_size
      return batch_x, batch_y

train, train_label, test, test_label, dev, dev_label, w2v, vocab_size = load_data()

args.vocab_size = vocab_size
args.vec_size = 300
args.max_sent = max_len
args.num_classes = len(user2idx)
print('max sent: ', args.max_sent)
print('vocab size: ', args.vocab_size)
print('vec size: ', args.vec_size)
print('num_classes: ', args.num_classes)

train, train_label = shuffle(train, train_label)

with tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()

# label = ['-1', 'earn', 'money-fx', 'trade', 'acq', 'grain', 'interest', 'crude', 'ship']
# label = map(str,label)
threshold = 0.5

X = tf.placeholder(tf.int32, [args.batch_size, args.max_sent], name="input_x")
y = tf.placeholder(tf.int32, [args.batch_size, args.num_classes], name="input_y")
is_training = tf.placeholder_with_default(False, shape=())
learning_rate = tf.placeholder(dtype='float32')
margin = tf.placeholder(shape=(),dtype='float32')

l2_loss = tf.constant(0.0)

# w2v = np.array(w2v,dtype=np.float32)
X_embedding = None

if args.embedding_type == 'rand':
    W1 = tf.Variable(tf.random_uniform([args.vocab_size, args.vec_size], -0.2, 0.2), trainable=True, name="Wemb")
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[..., tf.newaxis]
if args.embedding_type == 'static':
    W1 = tf.Variable(w2v, trainable=False)
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[...,tf.newaxis]
if args.embedding_type == 'nonstatic':
    W1 = tf.Variable(w2v, trainable = True)
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[..., tf.newaxis]
if args.embedding_type == 'multi-channel':
    W1 = tf.Variable(w2v, trainable = True)
    W2 = tf.Variable(w2v, trainable = False)
    X_1 = tf.nn.embedding_lookup(W1, X)
    X_2 = tf.nn.embedding_lookup(W2, X)
    X_1 = X_1[...,tf.newaxis]
    X_2 = X_2[...,tf.newaxis]
    X_embedding = tf.concat([X_1, X_2], axis=-1)

tf.logging.info("input dimension:{}".format(X_embedding.get_shape()))

poses, activations = None, None
if args.model_type == 'capsule-A':
    poses, activations = capsule_model_A(X_embedding, args.num_classes)
if args.model_type == 'capsule-B':
    poses, activations = capsule_model_B(X_embedding, args.num_classes)
if args.model_type == 'CNN':
    poses, activations = baseline_model_cnn(X_embedding, args.num_classes)
if args.model_type == 'KIMCNN':
    poses, activations = baseline_model_kimcnn(X_embedding, args.max_sent, args.num_classes)

loss = None
if args.loss_type == 'spread_loss':
    loss = spread_loss(y, activations, margin)
if args.loss_type == 'margin_loss':
    loss = margin_loss(y, activations)
if args.loss_type == 'cross_entropy':
    loss = cross_entropy(y, activations)
if args.loss_type == 'softmax':
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=activations)
    loss = tf.reduce_mean(loss)

y_pred = tf.argmax(activations, axis=1, name="y_proba")
correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
gradients, variables = zip(*optimizer.compute_gradients(loss))

grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
              for g in gradients if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

with tf.control_dependencies(grad_check):
    training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


sess = tf.InteractiveSession()
from keras import utils

n_iterations_per_epoch = len(train) // args.batch_size
n_iterations_test = len(test) // args.batch_size
n_iterations_dev = len(dev) // args.batch_size

mr_train = BatchGenerator(train, train_label, args.batch_size, 0)
mr_dev = BatchGenerator(dev, dev_label, args.batch_size, 0)
mr_test = BatchGenerator(test, test_label, args.batch_size, 0)

best_model = None
best_epoch = 0
best_acc_val = 0.

init = tf.global_variables_initializer()
sess.run(init)

variable_names = [v.name for v in tf.trainable_variables()]
print(variable_names)
lr = args.learning_rate
m = args.margin
for epoch in range(args.num_epochs):
    assert activations is not None and poses is not None
    for iteration in range(1, n_iterations_per_epoch + 1):
        X_batch, y_batch = mr_train.next()
        y_batch = utils.to_categorical(y_batch, args.num_classes)
        _, loss_train, probs, capsule_pose = sess.run(
            [training_op, loss, activations, poses],
            feed_dict={X: X_batch[:, :args.max_sent],
                       y: y_batch,
                       is_training: True,
                       learning_rate:lr,
                       margin:m})
        # print('capsule_pose: ', capsule_pose)
        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                  iteration, n_iterations_per_epoch,
                  iteration * 100 / n_iterations_per_epoch,
                  loss_train),
              end="")
    loss_vals, acc_vals = [], []
    for iteration in range(1, n_iterations_dev + 1):
        X_batch, y_batch = mr_dev.next()
        y_batch = utils.to_categorical(y_batch, args.num_classes)
        loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch[:,:args.max_sent],
                           y: y_batch,
                           is_training: False,
                           margin:m})
        loss_vals.append(loss_val)
        acc_vals.append(acc_val)
    loss_val, acc_val = np.mean(loss_vals), np.mean(acc_vals)
    print("\rEpoch: {}  Val accuracy: {:.1f}%  Loss: {:.4f}".format(
        epoch + 1, acc_val * 100, loss_val))
    if args.model_type == 'CNN' or args.model_type == 'KIMCNN':
        lr = max(1e-6, lr * 0.8)
    if args.loss_type == 'margin_loss':
        m = min(0.9, m + 0.1)

loss_tests, acc_tests = [], []
for iteration in range(1, n_iterations_test + 1):
    X_batch, y_batch = mr_test.next()
    y_batch = utils.to_categorical(y_batch, args.num_classes)
    loss_test, acc_test = sess.run(
        [loss, accuracy],
        feed_dict={X: X_batch[:, :args.max_sent],
                   y: y_batch,
                   is_training: False,
                   margin: m})
    loss_tests.append(loss_test)
    acc_tests.append(acc_test)
loss_test, acc_test = np.mean(loss_tests), np.mean(acc_tests)
print("test accuracy: {:.1f}%  Loss: {:.4f}".format(acc_test * 100, loss_test))

