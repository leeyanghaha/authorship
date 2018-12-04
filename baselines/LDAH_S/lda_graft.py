from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from baselines.LDAH_S import data

text = ["I am a history junkie, especially about the Gold Rushes in the US and Australia. " \
       "So I had great expectations that this book would add to my knowledge base - but " \
       "unfortunately it didn't add as much as some historical adventure novels I have " \
       "read in the same setting. This book was really no more than a long University " \
       "essay but did sum up a few essential facts about the Californian Gold Rush:* " \
       "Most went to the gold fields to get rich and go home and not to settle in " \
       "California.* Many never made it to California because the trip from the Eastern " \
       "States was monumental and fraught with dangers, on a wagon train " \
       "(with disease, hunger and Indians), round the treacherous Cape Horn or by sea and" \
       "land across the jungles of the Panama isthmus.* More people became rich " \
       "supplying the miners than the miners themselves.* About 1 in 5 miners died within " \
       "6 months of getting to the gold fields from accidents, disease and malnutrition.*" \
       "The Gold Rush was the trigger and the financial base for the construction of a " \
       "trans-continental railway that eventually brought settlers in great numbers to " \
       "the rich resources of California."]
        # "Peter James writes good UK police thrille but this one is not as gripping as "
        # "other police thrillers I have read recently. Interestingly, in the book Inspector"
        # " Grace says that he doesn't watch UK police thrillers on TV because they are not"
        # " very exciting and full of police procedurals and prefers US police thrillers "
        # "because they can do more and have more action. That comment is relevant to this "
        # "book until the last chapters.I liked the book (that's what 4 stars means) but "
        # "didn't love it. The police characters were very realistic, but the baddies were "
        # "a bit difficult to understand. In the early chapters I found things a bit tedious"
        # " but things warmed up towards the ending.The plot is intriguing - an out of "
        # "control stag party where a bridegroom's 4 friends bury him in a coffin to get "
        # "back at his practical jokes in the past. Things go really wrong when all 4 are "
        # "killed in a traffic accident and no-one appears to know what has happened to the "
        # "bridegroom. The bride and his business partner are extremely distressed.One major "
        # "hangup for me is the name of the main character. A friend has difficulty with "
        # "the Phryne (fry-knee) Fisher series by Kerry Greenwood because she can't cope with"
        # " the first name. Here I have difficulty with Detective Superintendent Roy Grace's "
        # "last name because it is also a female first name. Throughout the book the main "
        # "character is often just referred to as Grace and I kept thinking \"how did that "
        # "female character get involved",
        # "A sleepy little town is the perfect place to escape from it all. Even escaping "
        # "from an abusive ex. Lexie Meyers has rebuilt a life for herself in Star Harbor. "
        # "As chef and owner of her own small restaurant Lexi serves up low brow fare and"
        # "a side of spice.Sebastian Grayson is a chef on the rise of stardom. High brow "
        # "restaurants and television shows are his ticket. Seb also has a rep for being "
        # "one of the hottest catches around.Well from here on out you get the gist of it a"
        # "ll. They meet; sparks fly as Lexi doesn't fall at Sebs' feet. He is intrigued to "
        # "find a girl who serves him up on a platter cold. No his usual dish so to speak."
        # " (I know puns aplenty today) *HA!There is a bit of mystery when notes keep "
        # "appearing with a threatening overture. Lexi is attacked and her deliveries are "
        # "beginning to not show up.My thoughts on this debut title......4 Stars on the "
        # "love connection. It was cute and spunky. They had sizzle and Seb is very much "
        # "an alpha male. Always a good thing.2.5 Stars on the mystery/suspense. It was "
        # "easy to figure out what was going on from the beginning. I wish the author had"
        # " stuck to one plotline with this. There were too many little side twists that "
        # "in my opinion wasn't needed and some plots were left unfinished. It is possible"
        # " that this will be resolved in later books but it just felt incomplete in this"
        # " one.There was a bit of filler in the book as well. I didn't need all the "
        # "explanations' on cutting styles for chops and how the kitchen runs. I am sure "
        # "some will find it informative but for me it just took away from the story and "
        # "well...filler.I do see this series getting better and I did like the writing "
        # "style very much. Will I be looking out for the next book? Sure I will :)3 StarsT~"]

x = x.toarray()[0][:50].reshape(1, -1)
print(x)
print(x.shape)
lda = LDA(n_topics=20)
lda_res = lda.fit_transform(x)
# lda_res = lda.transform(lda_res)
print(type(lda_res))
a = lda_res[0, :]
b = lda_res[0, :]
print(data.Hellinger_distance(a, b))
# print(lda_res.split(' '))


