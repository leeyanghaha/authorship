import spacy
from spacy import displacy
import utils.key_utils as ku
import utils.review_data_utils as du
from collections import Counter
import torch
# nlp = spacy.load('en')
# doc = nlp("So I had great expectations that this book would add to my knowledge base - but "
#           "unfortunately it didn't add as much as some historical adventure novels I "
#           "have read in the same setting")
# sentences = list(doc.sents)
# displacy.serve(doc, style='dep')


if __name__ == '__main__':
    print(torch.tensor([0, 4]) * torch.tensor([2,3]))
