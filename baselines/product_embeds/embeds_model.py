import math
from .product_process import ProductVocab
import numpy as np
import time
from gensim.models import Word2Vec
from . import walker
import utils.key_utils as ku
import utils.function_utils as fu


class SeqSkipGramProduct:
    def __init__(self, reviews, product2idx):
        self.product_vocab = ProductVocab(reviews, product2idx)
        self.cur_matrix = self.product_vocab.products_adj_matrix()
        self.product_num = len(product2idx)
        self.neg_n = 0.01
        self.embeds_dim = 300
        self.product_embeds = self.product_embedding()

    def objective_func(self):
        sum = 0
        item1 = item2 = 0
        for i in range(self.cur_matrix.shape[0]):
            for j in range(self.cur_matrix.shape[1]):
                if i != j:
                    if self.cur_matrix[i, j] != 0:
                        n_p = self.cur_matrix[i, j]
                        log_s = math.log(self.sigmod(self.product_embeds[i, :],
                                                     self.product_embeds[j, :]))
                        item1 = n_p * log_s
                    else:
                        freq1, freq2 = self.product_vocab.get_product_freq(i, j)
                        n_m = self.neg_n * math.pow(freq1 * freq2, 0.75)
                        log_s = math.log(self.sigmod(-self.product_embeds[i, :],
                                                     self.product_embeds[j, :]))
                        item2 = n_m * log_s
                    sum += (item1 + item2)
        return sum


    def products_num(self):
        pn = 0
        for i in range(self.cur_matrix.shape[0]):
            pn += np.sum(self.cur_matrix[i, :])
        return pn

    def product_embedding(self):
        return 0.1 * np.ones(shape=(100, 300), dtype=np.int32)

    def sigmod(self, vk, vl):
        x = float(np.dot(vk, vl))
        if x < -10:
            return 0
        else:
            return 1 / (1 + math.exp(-x))

    def gradient_ascent(self, learning_rate, epochs):
        for idx in range(epochs):
            print('epoch {}: '.format(idx + 1))
            for i in range(self.cur_matrix.shape[0]):
                for j in range(self.cur_matrix.shape[0]):
                    if i != j:
                        self.product_embeds[i, :] += learning_rate * self.gradient(i, j, i)
                        self.product_embeds[j, :] += learning_rate * self.gradient(i, j, j)
            print(self.objective_func())


    def gradient(self, i, j, factor):
        vk = self.product_embeds[i, :]
        vl = self.product_embeds[j, :]
        item1 = item2 = 0
        factor_embeds = self.product_embeds[factor, :]
        if self.cur_matrix[i, j] != 0:
            n_p = self.cur_matrix[i, j]
            g_p = (1 - self.sigmod(vk, vl)) * factor_embeds
            item1 = n_p * g_p
        else:
            freq1, freq2 = self.product_vocab.get_product_freq(i, j)
            n_m = self.neg_n * math.pow(freq1 * freq2, 0.75)
            g_m = (1 - self.sigmod(-self.product_embeds[i, :], self.product_embeds[j, :])) * factor_embeds
            item2 = -n_m * g_m
        return item1 - item2

class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = walker.Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()


def node2vec_train(reviews, product2idx, param):
    product_vocab = ProductVocab(reviews, product2idx)
    product_vocab.read_edge_weights()
    model = Node2vec(graph=product_vocab, path_length=param['path_length'],
                     num_paths=param['num_paths'], dim=param['dim'], workers=param['workers'],
                     p=param['p'], q=param['q'], dw=param['dw'])
    print('Save embeddings....')
    model.save_embeddings(ku.products_embeds)


def load_node2vec_embedding(file, product_num, dim):
    array = fu.load_file(file)
    embeds = np.zeros(shape=(product_num, dim), dtype=np.float32)
    for i in range(1, len(array)):
        item = array[i].split(' ')
        idx = int(item[0])
        embedding = [float(i) for i in item[1: ]]
        embeds[idx, :] = embedding
    return embeds