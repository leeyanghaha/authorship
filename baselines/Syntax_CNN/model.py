from keras.layers import Dense, Conv2D, Input, Embedding, Dropout, Flatten,  \
    MaxPool2D, Reshape, concatenate, multiply
from keras.models import Model
import keras
from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import regularizers
import utils.key_utils as ku


class Net:
    def __init__(self, **params):
        self._check_input(**params)
        self.position_embedding = Embedding(input_dim=self.max_pos_num, output_dim=self.syntax_dim,
                                            input_length=self.max_pos_num*self.max_words_num)
        self.pos_type_embedding = Embedding(input_dim=self.pos_type_num, output_dim=self.syntax_dim,
                                            input_length=self.max_pos_num*self.max_words_num)
        self.ngram_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.ngram_dim,
                                         input_length=self.max_ngram_len)
        self.syntax_conv = Conv2D(self.filters, (self.kernel_size, self.syntax_dim), activation='relu', padding='valid')
        self.content_conv = Conv2D(self.filters, (self.kernel_size, self.ngram_dim), activation='relu', padding='valid')
        self.flatten = Flatten()
        self.dropout = Dropout(0.25)
        self.slicing_lambda = Lambda(lambda x:x[:])
        self.sum_lambda = Lambda(self.reduce_sum)
        self.model = self.net()
        self.compile()

    def _check_input(self, **params):
        self.max_words_num = params[ku.max_words_num]
        self.syntax_dim = params['syntax_dim']
        self.max_ngram_len = params[ku.max_ngram_len]
        self.ngram_dim = params['ngram_dim']
        self.pos_type_num = params['pos_type_num']
        self.out_dim = params['out_dim']
        self.max_pos_num = params['max_pos_num']
        self.vocab_size = params['vocab_size']
        self.filters = params['filters']
        self.kernel_size = params['kernel_size']
        self.batch_size = params['batch_size']
        self.loss = params['loss']

    def net(self):
        position_input = Input(shape=(self.max_words_num * self.max_pos_num, ), name='position_input')
        content_input = Input(shape=(self.max_ngram_len, ), name='content_input')
        pos_input = Input(shape=(self.max_words_num * self.max_pos_num, ), name='pos_input')

        position_embedding = self.position_embedding(position_input)
        # position_embedding = self.dropout(position_embedding)

        pos_embedding = self.pos_type_embedding(pos_input)
        # pos_embedding = self.dropout(pos_embedding)

        content_embedding = self.ngram_embedding(content_input)
        content_embedding = self.dropout(content_embedding)
        content_embedding = Reshape((content_embedding.shape[1], content_embedding.shape[2], 1))(content_embedding)

        multiply_pos_position = multiply([position_embedding, pos_embedding])
        multiply_pos_position = Reshape((self.max_words_num, self.max_pos_num, self.syntax_dim))(multiply_pos_position)


        syntax_embedding = self.sum_lambda(multiply_pos_position)
        syntax_embedding = self.dropout(syntax_embedding)
        syntax_embedding = Reshape((syntax_embedding.shape[1], syntax_embedding.shape[2], 1))(syntax_embedding)


        syntax_conv = self.syntax_conv(syntax_embedding)
        max_pool_syntax = MaxPool2D((syntax_conv.shape[1], 1))(syntax_conv)
        max_pool_syntax = self.slicing_lambda(max_pool_syntax)

        content_conv = self.content_conv(content_embedding)
        max_pool_content = MaxPool2D((content_conv.shape[1], 1))(content_conv)
        max_pool_content = self.slicing_lambda(max_pool_content)

        concat = concatenate([max_pool_content, max_pool_syntax], axis=3)
        concat = self.flatten(concat)

        out = Dense(self.out_dim, activation='softmax', kernel_regularizer=regularizers.l2(0.01)
                    )(concat)
        model = Model(inputs=[content_input, pos_input, position_input], outputs=out)
        return model

    def compile(self):
        optimizer = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

    def fit(self, train_x, train_y):
        train_y = to_categorical(train_y, num_classes=self.out_dim)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.001,
                                       mode='max')
        self.model.fit(train_x, train_y, validation_split=0.2, batch_size=self.batch_size,
                       callbacks=[early_stopping], epochs=100)

    def reduce_sum(self, x):
        return K.sum(x, axis=2)

    def save_weight(self, path):
        self.model.save_weights(path)

    def evaluate(self, x, y):
        y = to_categorical(y, num_classes=self.out_dim)
        return self.model.evaluate(x, y, batch_size=self.batch_size)

    def load_weight(self, path):
        self.model.load_weights(path)


