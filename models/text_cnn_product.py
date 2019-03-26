from keras.layers import Dense, Conv2D, Input, Embedding, Dropout, Flatten,  \
    MaxPool2D, Reshape, BatchNormalization, ReLU
from keras.models import Model
import keras
from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Layer
from keras.backend import tensorflow_backend as K
# import keras.backend as K
import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import TensorBoard
from keras.initializers import Constant


class TextCNNPro:
    def __init__(self, **kwargs):
        self.check_input_error(**kwargs)

        self.ngram_embeds = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                      input_length=self.max_ngram_len)
        self.slicing_lambda = Lambda(lambda x: x[:])
        self.flatten = Flatten()
        self.default_loss = 'categorical_crossentropy'
        if self.method == 1:
            self.model = self.net1()
            self.compile()
        else:
            self.model = self.net2()
            self.compile()

    def check_input_error(self, **param):
        assert len(param['kernel_size']) == 3
        # assert param['product_embedding_dim'] == 3 * param['feature_num']
        self.kernel_size = param['kernel_size']
        self.batch_size = param['batch_size']
        self.epochs = param['epochs']
        self.embedding_dim = param['embedding_dim']
        self.user_num = param['user_num']
        self.max_ngram_len = param['max_ngram_len']
        self.feature_num = param['feature_num']
        self.vocab_size = param['vocab_size']
        self.product_num = param['product_num']
        self.method = param['method']
        self.product_embedding_dim = param['product_embedding_dim']
        self.pre_trained_embeds = param['pre_trained_embeds']


    def net1(self):
        # method 1: 一个product embedding, later fusion.
        text_input = Input(shape=(self.max_ngram_len,), name='text_input')
        product_input = Input(shape=(1,), name='product_input')
        text_embeds = self.ngram_embeds(text_input)
        text_embeds = Reshape((text_embeds.shape[1], text_embeds.shape[2], 1))(text_embeds)
        if self.pre_trained_embeds is None:
            self.product_embeds = Embedding(input_dim=self.product_num, output_dim=self.embedding_dim,
                                        input_length=1, name='product_embedding')
        else:
            self.product_embeds = Embedding(input_dim=self.product_num, input_length=1,
                                            embeddings_initializer=Constant(self.pre_trained_embeds),
                                            trainable=True, output_dim=self.embedding_dim,
                                            name='product_embedding')
        product_embeds = self.product_embeds(product_input)


        conv1 = Conv2D(self.feature_num, (self.kernel_size[0], self.embedding_dim), padding='valid')(text_embeds)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        max_pool1 = MaxPool2D((conv1.shape[1], 1))(conv1)


        conv2 = Conv2D(self.feature_num, (self.kernel_size[1], self.embedding_dim), padding='valid')(text_embeds)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        max_pool2 = MaxPool2D((conv2.shape[1], 1))(conv2)


        conv3 = Conv2D(self.feature_num, (self.kernel_size[2], self.embedding_dim), padding='valid')(text_embeds)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        max_pool3 = MaxPool2D((conv3.shape[1], 1))(conv3)


        max_pool1 = self.slicing_lambda(max_pool1)
        max_pool2 = self.slicing_lambda(max_pool2)
        max_pool3 = self.slicing_lambda(max_pool3)
        product_embeds = Reshape((1, 1, product_embeds.shape[2]))(product_embeds)
        product_embeds = self.slicing_lambda(product_embeds)


        concat = keras.layers.concatenate([max_pool1, max_pool2, max_pool3, product_embeds])

        concat = self.flatten(concat)
        concat = Dropout(0.5)(concat)
        concat = Dense(500)(concat)
        concat = Dropout(0.5)(concat)


        out = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01)
                    )(concat)
        self.loss = self.default_loss
        model = Model(inputs=[text_input, product_input], outputs=out)
        return model

    def net2(self):
        text_input = Input(shape=(self.max_ngram_len,), name='text_input')
        product_input = Input(shape=(1,), name='product_input')
        text_embeds = self.ngram_embeds(text_input)
        text_embeds = Reshape((text_embeds.shape[1], text_embeds.shape[2], 1))(text_embeds)
        if self.pre_trained_embeds is None:
            self.product_embeds = Embedding(input_dim=self.product_num, output_dim=self.embedding_dim,
                                            input_length=1, name='product_embedding')
        else:
            self.product_embeds = Embedding(input_dim=self.product_num, input_length=1,
                                            embeddings_initializer=Constant(self.pre_trained_embeds),
                                            trainable=True, output_dim=self.embedding_dim,
                                            name='product_embedding')
        product_embeds = self.product_embeds(product_input)

        conv1 = Conv2D(self.feature_num, (self.kernel_size[0], self.embedding_dim), padding='valid')(text_embeds)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        max_pool1 = MaxPool2D((conv1.shape[1], 1))(conv1)

        conv2 = Conv2D(self.feature_num, (self.kernel_size[1], self.embedding_dim), padding='valid')(text_embeds)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        max_pool2 = MaxPool2D((conv2.shape[1], 1))(conv2)

        conv3 = Conv2D(self.feature_num, (self.kernel_size[2], self.embedding_dim), padding='valid')(text_embeds)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        max_pool3 = MaxPool2D((conv3.shape[1], 1))(conv3)

        max_pool1 = self.slicing_lambda(max_pool1)
        max_pool2 = self.slicing_lambda(max_pool2)
        max_pool3 = self.slicing_lambda(max_pool3)
        product_embeds = Reshape((1, 1, product_embeds.shape[2]))(product_embeds)
        product_model = self.slicing_lambda(product_embeds)

        text_model = keras.layers.concatenate([max_pool1, max_pool2, max_pool3])

        text_product_model = keras.layers.concatenate([max_pool1, max_pool2, max_pool3, product_model])


    def net2(self):
        # 2 个 embedding, later fusion, KL loss added.
        text_input = Input(shape=(self.max_ngram_len, ), name='text_input')
        product_input = Input(shape=(1, ), name='product_input')
        text_embeds = self.ngram_embeds(text_input)
        self.product_embeds = Embedding(input_dim=self.product_num, output_dim=self.product_embedding_dim,
                                        input_length=1)
        product_embeds1 = self.product_embeds(product_input)
        product_embeds2 = self.product_embeds(product_input)
        text_embeds = Dropout(0.25)(text_embeds)
        text_embeds = Reshape((text_embeds.shape[1], text_embeds.shape[2], 1))(text_embeds)  # Conv2d 需要channel值
        conv1 = Conv2D(self.feature_num, (self.kernel_size[0], self.embedding_dim), activation='relu', padding='valid')(
            text_embeds)
        max_pool1 = MaxPool2D((conv1.shape[1], 1))(conv1)
        conv2 = Conv2D(self.feature_num, (self.kernel_size[1], self.embedding_dim), activation='relu', padding='valid')(
            text_embeds)
        max_pool2 = MaxPool2D((conv2.shape[1], 1))(conv2)
        conv3 = Conv2D(self.feature_num, (self.kernel_size[2], self.embedding_dim), activation='relu', padding='valid')(
            text_embeds)
        max_pool3 = MaxPool2D((conv3.shape[1], 1))(conv3)
        max_pool1 = self.slicing_lambda(max_pool1)
        max_pool2 = self.slicing_lambda(max_pool2)
        max_pool3 = self.slicing_lambda(max_pool3)
        product_embeds1 = Reshape((1, 1, product_embeds1.shape[2]))(product_embeds1)
        product_embeds2 = Reshape((1, 1, product_embeds2.shape[2]))(product_embeds2)
        product_embeds1 = self.flatten(self.slicing_lambda(product_embeds1))
        product_embeds2 = self.flatten(self.slicing_lambda(product_embeds2))
        text_out = keras.layers.concatenate([max_pool1, max_pool2, max_pool3])
        text_out = self.flatten(text_out)
        self.loss = self.KL_loss(product_embeds1, product_embeds2, text_out)
        concat = keras.layers.concatenate([text_out, product_embeds1, product_embeds2])
        concat = Dropout(0.25)(concat)
        out = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01)
                    )(concat)
        model = Model(inputs=[text_input, product_input], outputs=out)
        return model

    def net3(self):
        pass

    def compile(self):
        optimizer = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

    def fit(self, train_x, train_y):
        train_y = to_categorical(train_y, num_classes=self.user_num)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.0001,
                                       mode='max')
        tbcallbacks = TensorBoard(log_dir='./product_logs', histogram_freq=0, write_graph=True, write_images=True,
                                  )
        self.model.fit(train_x, train_y, validation_split=0.2, batch_size=self.batch_size,
                       callbacks=[early_stopping, tbcallbacks], epochs=self.epochs)

    def save_weight(self, path):
        self.model.save_weights(path)

    def evaluate(self, x, y):
        y = to_categorical(y, num_classes=self.user_num)
        return self.model.evaluate(x, y, batch_size=self.batch_size)

    def load_weight(self, path):
        self.model.load_weights(path)

    def kld(self, E, P):
        E = K.clip(E, K.epsilon(), 1)
        P = K.clip(P, K.epsilon(), 1)
        return K.sum(P * K.log(P / E), axis=-1)

    def _my_loss(self, y_true, y_pred, E1, E2, P):
        predict_loss = K.categorical_crossentropy(y_true, y_pred)
        similar_loss = self.kld(E1, P)
        disimilar_loss = self.kld(E2, P)
        return predict_loss + similar_loss - disimilar_loss

    def KL_loss(self,E1, E2, P):
        def dice(y_true, y_pred):
            return self._my_loss(y_true, y_pred, E1, E2, P)
        return dice

    def get_product_embedding(self, product_id):
        layer_name = 'product_embedding'
        product_embedding_model = Model(inputs=self.model.input,
                                        outputs=self.model.get_layer(layer_name).output)
        out = product_embedding_model.predict(product_id)
        return out


class SelfRegulationPro:
    def __init__(self, **param):
        self.check_input_error(**param)
        self.ngram_embeds = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                      input_length=self.max_ngram_len)
        self.slicing_lambda = Lambda(lambda x: x[:])
        self.flatten = Flatten()
        self.default_loss = 'categorical_crossentropy'
        self.model = self.net()
        self.compile()

    def check_input_error(self, **param):
        assert len(param['kernel_size']) == 3
        self.kernel_size = param['kernel_size']
        self.batch_size = param['batch_size']
        self.epochs = param['epochs']
        self.embedding_dim = param['embedding_dim']
        self.user_num = param['user_num']
        self.max_ngram_len = param['max_ngram_len']
        self.feature_num = param['feature_num']
        self.vocab_size = param['vocab_size']
        self.product_num = param['product_num']
        self.product_embedding_dim = param['product_embedding_dim']

    def net(self):
        text_input = Input(shape=(self.max_ngram_len, ), name='text_input')
        product_input = Input(shape=(1, ), name='product_input')
        text_embeds = self.ngram_embeds(text_input)
        product_embeds1 = Embedding(input_dim=self.product_num, output_dim=self.product_embedding_dim,
                                    input_length=1, name='product_embeds1')
        product_embeds2 = Embedding(input_dim=self.product_num, output_dim=self.product_embedding_dim,
                                    input_length=1, name='product_embeds2')
        product_embeds1 = product_embeds1(product_input)
        product_embeds2 = product_embeds2(product_input)



        product_embeds1 = Reshape((1, 1, product_embeds1.shape[2]))(product_embeds1)
        product_embeds2 = Reshape((1, 1, product_embeds2.shape[2]))(product_embeds2)
        product_embeds1 = self.flatten(self.slicing_lambda(product_embeds1))
        product_embeds2 = self.flatten(self.slicing_lambda(product_embeds2))
        product_embeds1 = Dropout(0.25)(product_embeds1)
        product_embeds2 = Dropout(0.25)(product_embeds2)


        text_embeds = Dropout(0.25)(text_embeds)
        text_embeds = Reshape((text_embeds.shape[1], text_embeds.shape[2], 1))(text_embeds)  # Conv2d 需要channel值
        conv1 = Conv2D(self.feature_num, (self.kernel_size[0], self.embedding_dim), activation='relu', padding='valid')(
            text_embeds)
        max_pool1 = MaxPool2D((conv1.shape[1], 1))(conv1)
        conv2 = Conv2D(self.feature_num, (self.kernel_size[1], self.embedding_dim), activation='relu', padding='valid')(
            text_embeds)
        max_pool2 = MaxPool2D((conv2.shape[1], 1))(conv2)
        conv3 = Conv2D(self.feature_num, (self.kernel_size[2], self.embedding_dim), activation='relu', padding='valid')(
            text_embeds)
        max_pool3 = MaxPool2D((conv3.shape[1], 1))(conv3)
        max_pool1 = self.slicing_lambda(max_pool1)
        max_pool2 = self.slicing_lambda(max_pool2)
        max_pool3 = self.slicing_lambda(max_pool3)
        text_out = self.flatten(keras.layers.concatenate([max_pool1, max_pool2, max_pool3]))

        o_g_concat = keras.layers.concatenate([text_out, product_embeds1])
        print('o_g_concat.shape: ', o_g_concat.shape)
        o_g = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                    name='o_g')(o_g_concat)
        self.o_g_losss = self.default_loss
        o_g_f_concat = keras.layers.concatenate([text_out, product_embeds2])
        o_g_f = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      name='o_g_f')(o_g_f_concat)
        self.o_g_f_losss = self.o_g_f_loss()


        d_ = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.1),
                    name='d_')(o_g_concat)
        self.d_losss = self.d_loss(product_embeds1, product_embeds2)
        d_f = Dense(self.user_num, activation='softmax', kernel_regularizer=regularizers.l2(0.1),
                      name='d_f')(o_g_f_concat)
        self.d_f_losss = self.default_loss
        model = Model(inputs=[text_input, product_input], outputs=[o_g, o_g_f, d_, d_f])
        model.summary()
        return model

    def kld(self, product_embeds1, product_embeds2):
        product_embeds2 = K.clip(product_embeds2, K.epsilon(), 1)
        product_embeds1 = K.clip(product_embeds1, K.epsilon(), 1)
        return K.sum(product_embeds1 * K.log(product_embeds1 / product_embeds2), axis=-1)

    def _my_loss(self, y_true, y_pred, product_embeds1, product_embeds2):
        predict_loss = K.categorical_crossentropy(y_true, y_pred)
        similar_loss = self.kld(product_embeds1, product_embeds2)
        return predict_loss - 0.2 * similar_loss

    def d_loss(self, product_embeds1, product_embeds_2):
        def my_loss(y_true, y_pred):
            return self._my_loss(y_true, y_pred, product_embeds1, product_embeds_2)
        return my_loss

    def o_g_f_loss(self):
        def dice(y_true, y_pred):
            return - K.categorical_crossentropy(y_true, y_pred)
        return dice

    def compile(self):
        optimizer = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss={'o_g': self.o_g_f_losss,
                                                      'o_g_f': self.o_g_f_losss,
                                                      'd_': self.d_losss,
                                                      'd_f': self.d_f_losss},
                           metrics=['accuracy'])

    def fit(self, inputs, outputs):
        train_y = to_categorical(outputs['d_'], num_classes=self.user_num)
        for output_name in outputs:
            outputs[output_name] = train_y
        early_stopping = EarlyStopping(monitor='val_d__acc', patience=5, min_delta=0.0001,
                                       mode='max')
        self.model.fit(inputs, outputs, validation_split=0.2, batch_size=self.batch_size,
                       callbacks=[early_stopping], epochs=self.epochs)

    def evaluate(self, inputs, y_true):
        text_input = inputs['text_input']
        product_input = inputs['product_input']
        layer_name = 'd_'
        layer_model = Model(inputs=self.model.input,
                            outputs=self.model.get_layer(layer_name).output)
        layer_out = layer_model.predict([text_input, product_input])
        y_true = list(y_true)
        y_pred = list()
        for i in range(layer_out.shape[0]):
            y_pred.append(np.argmax(layer_out[i, :]))
        return accuracy_score(y_true, y_pred)
