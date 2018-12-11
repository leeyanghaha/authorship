import os

# review item field
reviewer_name = 'reviewerName'
reviewer_ID = 'reviewerID'
reviewer_count = 'reviewerCount'
review_text = 'reviewText'
asin = 'asin'


# twitter field
twitter_id_str = 'id_str'
twitter_text = 'text'
twitter_user = 'user'
retweet = 'retweeted'
entities = 'entities'
media = 'media'
media_id_str = 'id_str'
image = 'image'


# review domain field
Cd = 'reviews_CDs_and_Vinyl_5'
Kindle = 'reviews_Kindle_Store'
Movie = 'reviews_Movies_and_TV'
Elect = 'reviews_Electronics'
Apps = 'reviews_Apps_for_Android'


#vocabulary filed
charngram2idx = 'charngram2idx'
wordngram2idx = 'wordngram2idx'
pos2idx = 'pos2idx'
UNK = 'UNK'
PAD = 0
root = '/home/yangl/research/authorship/'
user_root = os.path.join(root, 'data/user')
voca_root = os.path.join(root, 'model/vocabulary')
twitter_data_root = os.path.join(root, 'data/twitter/')
twitter_data_all = os.path.join(twitter_data_root, 'all')
twitter_process = os.path.join(twitter_data_root, 'process')
photo_dir = os.path.join(root, 'data/twitter/photo')


#model filed
model_root = os.path.join(root, 'model')
AA_of_MM = 'aa_of_mm'
CNN_AST_model = os.path.join(model_root, 'CNN_AST/weights.h5')
Syntax_CNN_model = os.path.join(model_root, 'Syntax_CNN/weights.h5')


# processing field
text_id = 'text_id'
user_id = 'user_id'
word_id = 'word_id'
ngram_id = 'ngram_id'
pos_id = 'pos_id'
pos_order_id = 'pos_order_id'


# data type field

review = 'reviews'
twitter = 'twitter'