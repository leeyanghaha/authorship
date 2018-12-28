import numpy as np
import utils.key_utils as ku
from collections import Counter
import networkx as nx
import utils.function_utils as fu


class ProductVocab:
    def __init__(self, reviews, product2idx):
        self.reviews = reviews
        self.product2idx = product2idx
        self.G = nx.DiGraph()
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def get_user_products(self):
        res = {}
        for review in self.reviews:
            user = review[ku.reviewer_ID]
            if user in res:
                res[user].append(self.product2idx[review[ku.asin]])
            else:
                res.update({user: [self.product2idx[review[ku.asin]]]})
        return res

    def products_adj_matrix(self):
        adj_matrix = np.zeros((len(self.product2idx), len(self.product2idx)), dtype=np.int32)
        user_products = self.get_user_products()
        for user, products in user_products.items():
            for i in range(len(products) - 1):
                adj_matrix[products[i], products[i+1]] += 1
                adj_matrix[products[i+1], products[i]] += 1
        return adj_matrix

    def get_product_freq(self, product1, product2):
        user_products = self.get_user_products()
        counter = Counter()
        for user, products in user_products.items():
            counter.update(products)
        total = sum(counter.values())
        freq1 = counter[product1] / total
        freq2 = counter[product2] / total
        return freq1, freq2

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_edge_weights(self):
        weighted_adj_matrix = self.products_adj_matrix()
        res = []
        for i in range(weighted_adj_matrix.shape[0]):
            for j in range(i+1, weighted_adj_matrix.shape[1]):
                if weighted_adj_matrix[i, j] != 0:
                    res.append([str(i), str(j), 0.5 * weighted_adj_matrix[i][j]])
        for item in res:
            src, dst, w = item[0], item[1], item[2]
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = float(w)
            self.G[dst][src]['weight'] = float(w)
        self.encode_node()

    def load_products_embeds(self, file, product_num, dim):
        array = fu.load_file(file)
        embeds = np.zeros(shape=(product_num, dim), dtype=np.float32)
        for item in array:
            item = item.split(' ')
            idx = int(item[0])
            embedding = [float(i) for i in item[1: ]]
            embeds[idx, :] = embedding
        return embeds



