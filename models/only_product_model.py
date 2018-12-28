from keras.layers import Embedding, Input, Dense, Flatten, Conv2D, \
    MaxPool2D, Reshape, BatchNormalization, ReLU
from keras.models import Model
import keras
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
import numpy as np

class OnlyProduct:
    def __init__(self, **param):
        self.embedding_dim = param['embedding_dim']
        self.user_num = param['user_num']
        self.product_num = param['product_num']
        self.loss = 'categorical_crossentropy'
        self.batch_size = param['batch_size']
        self.pre_train_embeds = param['pre_train_embeds']
        self.epochs = 500
        self.model = self.net()
        self.compile()

    def net(self):
        product_input = Input(shape=(1,), name='product_input')
        if self.pre_train_embeds is None:
            embeds = Embedding(input_dim=self.product_num, input_length=1, output_dim=300)(product_input)
        else:
            embeds = Embedding(input_dim=self.product_num, input_length=1, embeddings_initializer=Constant(self.pre_train_embeds),
                               trainable=False, output_dim=300)(product_input)
        embeds = Flatten()(embeds)
        out = Dense(self.user_num, activation='softmax')(embeds)

        model = Model(inputs=product_input, outputs=out)
        model.summary()
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

    def evaluate(self, x, y):
        y = to_categorical(y, num_classes=self.user_num)
        return self.model.evaluate(x, y, batch_size=self.batch_size)

    def predict(self, x_test):
        predict_prob = self.model.predict(x_test, batch_size=self.batch_size)
        res = []
        for i in range(predict_prob.shape[0]):
            res.append(np.argmax(predict_prob[i, :]))
        return res