from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
import utils.data_utils as du
import utils.function_utils as fu
import sklearn.utils as sku
import os

data_root = '/home/nfs/yangl/research/authorship/data/glue/review'
datahelper = du.DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = du.UserHelper()


reviews  = du.ReviewLoader(ku.Movie, product_num=100).get_data()

reviews = sku.shuffle(reviews)

input_file = os.path.join(data_root, 'input.txt')

def tokenize(review_text):
    import spacy
    nlp = spacy.load('en')
    doc = nlp(review_text)
    res = []
    for sent in doc.sents:
        res.append(sent.text)
    return res

def get_input(reviews):
    res = []
    with open(input_file, 'a') as f:
        for review in reviews:
            sentences = tokenize(review[ku.review_text])
            for sent in sentences:
                f.write(sent + '\n')
    return res
all = get_input(reviews)
# fu.dump_file(all, input_file)


