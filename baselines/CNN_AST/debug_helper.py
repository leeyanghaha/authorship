
ngram = 4
def recover(text_id, user_id, user2idx, ngram2idx):
    idx2u = idx2user(user2idx)
    user = idx2u[user_id[0]]
    idx2n = id2ngram(ngram2idx)
    text = ''
    for id in text_id[0]:
        ch = idx2n[id][0]
        text += ch
    return text, user


def idx2user(user2idx):
    idx = {}
    for user, id in user2idx.items():
        idx.update({id: user})
    return idx

def id2ngram(ngram2idx):
    idx = {}
    for ngram, id in ngram2idx.items():
        idx.update({id: ngram})
    return idx


