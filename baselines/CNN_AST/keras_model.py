from keras.layers import Dense, Conv2D, Input, Embedding, Dropout, Flatten,  MaxPool2D, Reshape
from keras.models import Model
import keras
from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import utils.key_utils as ku
import os

m = 300
w = [4, 7, 9]
batch_size = 32
epochs = 100

loss = 'categorical_crossentropy'
class Net:
    def __init__(self, vocab_size, max_ngram_num, embedding_dim, out_dim):
        self.vocab_size = vocab_size
        self.max_ngram_num = max_ngram_num
        self.user_num = out_dim
        self.embedding_dim = embedding_dim
        self.ngram_embeds = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                      input_length=max_ngram_num)
        self.slicing_lambda = Lambda(lambda x:x[:])
        self.flatten = Flatten()
        self.model = self.net()
        self.compile()

    def net(self):
        input = Input(shape=(self.max_ngram_num, ))
        embeds = self.ngram_embeds(input)
        embeds = Dropout(0.25)(embeds)
        embeds = Reshape((embeds.shape[1], embeds.shape[2], 1))(embeds) #Conv2d 需要channel值
        conv1 = Conv2D(m, (w[0], self.embedding_dim), activation='relu', padding='valid')(embeds)
        max_pool1 = MaxPool2D((conv1.shape[1], 1))(conv1)
        conv2 = Conv2D(m, (w[1], self.embedding_dim), activation='relu', padding='valid')(embeds)
        max_pool2 = MaxPool2D((conv2.shape[1], 1))(conv2)
        conv3 = Conv2D(m, (w[2], self.embedding_dim), activation='relu', padding='valid')(embeds)
        max_pool3 = MaxPool2D((conv3.shape[1], 1))(conv3)
        max_pool1 = self.slicing_lambda(max_pool1) #解决concatenate unhashbel type error 问题
        max_pool2 = self.slicing_lambda(max_pool2)
        max_pool3 = self.slicing_lambda(max_pool3)
        concat = keras.layers.concatenate([max_pool1, max_pool2, max_pool3])
        concat = self.flatten(concat)
        out = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01)
                    )(concat)
        model = Model(inputs=input, outputs=out)
        return model


    def compile(self):
        optimizer = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


    def fit(self, train_x, train_y):
        train_y = to_categorical(train_y, num_classes=self.user_num)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.0001,
                                       mode='max')

        self.model.fit(train_x, train_y, validation_split=0.2, batch_size=batch_size,
                       callbacks=[early_stopping], epochs=epochs)

    def save_weight(self, path):
        self.model.save_weights(path)

    def evaluate(self, x, y):
        y = to_categorical(y, num_classes=self.user_num)
        return self.model.evaluate(x, y, batch_size=batch_size)

    def load_weight(self, path):
        self.model.load_weights(path)





if __name__ == '__main__':
    print(keras.__version__)


