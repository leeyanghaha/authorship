from keras.layers import Dense, Conv2D, Input, Embedding, Dropout, Flatten,  MaxPool2D, Reshape
from keras.models import Model
import keras
from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers


class TextCNN:
    def __init__(self, **kwargs):
        self.check_input_error(**kwargs)
        self.ngram_embeds = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                      input_length=self.max_ngram_len)
        self.slicing_lambda = Lambda(lambda x:x[:])
        self.flatten = Flatten()
        self.model = self.net()
        self.compile()

    def check_input_error(self, **param):
        assert len(param['kernel_size']) == 3
        self.kernel_size = param['kernel_size']
        self.batch_size = param['batch_size']
        self.epochs = param['epochs']
        self.loss = param['loss']
        self.embedding_dim = param['embedding_dim']
        self.user_num = param['user_num']
        self.max_ngram_len = param['max_ngram_len']
        self.feature_num = param['feature_num']
        self.vocab_size = param['vocab_size']

    def net(self):
        input = Input(shape=(self.max_ngram_len, ))
        embeds = self.ngram_embeds(input)
        embeds = Dropout(0.25)(embeds)
        embeds = Reshape((embeds.shape[1], embeds.shape[2], 1))(embeds) #Conv2d 需要channel值
        conv1 = Conv2D(self.feature_num, (self.kernel_size[0], self.embedding_dim), activation='relu', padding='valid')(embeds)
        max_pool1 = MaxPool2D((conv1.shape[1], 1))(conv1)
        conv2 = Conv2D(self.feature_num, (self.kernel_size[1], self.embedding_dim), activation='relu', padding='valid')(embeds)
        max_pool2 = MaxPool2D((conv2.shape[1], 1))(conv2)
        conv3 = Conv2D(self.feature_num, (self.kernel_size[2], self.embedding_dim), activation='relu', padding='valid')(embeds)
        max_pool3 = MaxPool2D((conv3.shape[1], 1))(conv3)
        max_pool1 = self.slicing_lambda(max_pool1) #解决concatenate unhashbel type error 问题
        max_pool2 = self.slicing_lambda(max_pool2)
        max_pool3 = self.slicing_lambda(max_pool3)
        concat = keras.layers.concatenate([max_pool1, max_pool2, max_pool3])
        concat = Dropout(0.25)(concat)
        concat = self.flatten(concat)
        out = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01)
                    )(concat)
        model = Model(inputs=input, outputs=out)
        return model


    def compile(self):
        optimizer = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])


    def fit(self, train_x, train_y):
        train_y = to_categorical(train_y, num_classes=self.user_num)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.0001,
                                       mode='max')

        self.model.fit(train_x, train_y, validation_split=0.2, batch_size=self.batch_size,
                       callbacks=[early_stopping], epochs=self.epochs)

    def save_weight(self, path):
        self.model.save_weights(path)

    def evaluate(self, x, y):
        y = to_categorical(y, num_classes=self.user_num)
        return self.model.evaluate(x, y, batch_size=self.batch_size)

    def load_weight(self, path):
        self.model.load_weights(path)
