from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

# --------------------------------------------------------
class TextPreprocessor():
    def __init__(
        self,
    ) -> None:
        self.vectorizer = CountVectorizer()
        
    def preprocess(self, sentences: list, transform_only: bool = False) -> list:
        preprocessed = [sentence.lower() for sentence in sentences]
        
        
        # if transform_only:
        #     return self.vectorizer.transform(preprocessed)
        # return self.vectorizer.fit_transform(preprocessed)
        return preprocessed
    
    

# --------------------------------------------------------
with open('data/positive-reviews.txt') as infile:
    positive_lst = infile.read().split('\n')
    
with open('data/negative-reviews.txt') as infile:
    negative_lst = infile.read().split('\n')
    
with open('data/positive-words.txt') as infile:
    positive_words = infile.read().split('\n')
    
with open('data/negative-words.txt', encoding='latin-1') as infile:
    negative_words = infile.read().split('\n')

training_pos_size = int(len(positive_lst) * 0.8)
training_neg_size = int(len(positive_lst) * 0.8)

training_lst = positive_lst[:training_pos_size] + negative_lst[:training_neg_size]
testing_lst = positive_lst[training_pos_size:] + negative_lst[training_pos_size:]

training_pos_df = pd.DataFrame({
    'review': positive_lst[:training_pos_size],
    'sentiment': [1 for x in range(0, training_pos_size)]
})

test_pos_df = pd.DataFrame({
    'review': positive_lst[training_pos_size:],
    'sentiment': [1 for x in range(0, len(positive_lst) - training_pos_size)]
})

# feature extraction
# x1: count positive word
# x2: count negative word
# x3 1 if no in doc, else 0
# x4 count(1st and 2nd person pronoun)
# x5 1 if ! in doc, else 0
# x6 log(word_count)

def get_count_positive_words(text):
    return len([word for word in text.split(' ') if word in positive_words])

def get_count_negative_words(text):
    return len([word for word in text.split(' ') if word in negative_words])

def get_no(text):
    return 1 if 'no' in text.split(' ') else 0

def get_pronoun(text):
    return 0

def get_exclamation_mark(text):
    return 1 if '!' in text.split(' ') else 0

import numpy as np
def get_log_word_count(text: str):
    return np.log(len(text.split(' ')))

#
training_pos_df['x1'] = training_pos_df['review'].apply(get_count_positive_words)
training_pos_df['x2'] = training_pos_df['review'].apply(get_count_negative_words)
training_pos_df['x3'] = training_pos_df['review'].apply(get_no)
training_pos_df['x4'] = training_pos_df['review'].apply(get_pronoun)
training_pos_df['x5'] = training_pos_df['review'].apply(get_exclamation_mark)
training_pos_df['x6'] = training_pos_df['review'].apply(get_log_word_count)

# test set
test_pos_df['x1'] = test_pos_df['review'].apply(get_count_positive_words)
test_pos_df['x2'] = test_pos_df['review'].apply(get_count_negative_words)
test_pos_df['x3'] = test_pos_df['review'].apply(get_no)
test_pos_df['x4'] = test_pos_df['review'].apply(get_pronoun)
test_pos_df['x5'] = test_pos_df['review'].apply(get_exclamation_mark)
test_pos_df['x6'] = test_pos_df['review'].apply(get_log_word_count)

# -------------- negative review ------------
training_neg_df = pd.DataFrame({
    'review': negative_lst[:training_neg_size],
    'sentiment': [0 for x in range(0, training_neg_size)]
})

test_neg_df = pd.DataFrame({
    'review': negative_lst[training_neg_size:],
    'sentiment': [0 for x in range(0, len(negative_lst) - training_neg_size)]
})

training_neg_df['x1'] = training_neg_df['review'].apply(get_count_positive_words)
training_neg_df['x2'] = training_neg_df['review'].apply(get_count_negative_words)
training_neg_df['x3'] = training_neg_df['review'].apply(get_no)
training_neg_df['x4'] = training_pos_df['review'].apply(get_pronoun)
training_neg_df['x5'] = training_neg_df['review'].apply(get_exclamation_mark)
training_neg_df['x6'] = training_neg_df['review'].apply(get_log_word_count)

test_neg_df['x1'] = test_neg_df['review'].apply(get_count_positive_words)
test_neg_df['x2'] = test_neg_df['review'].apply(get_count_negative_words)
test_neg_df['x3'] = test_neg_df['review'].apply(get_no)
test_neg_df['x4'] = test_pos_df['review'].apply(get_pronoun)
test_neg_df['x5'] = test_neg_df['review'].apply(get_exclamation_mark)
test_neg_df['x6'] = test_neg_df['review'].apply(get_log_word_count)

# ----------- combine ---------------

training_df = pd.concat([
    training_pos_df,
    training_neg_df,
], axis=0, ignore_index=True)

test_df = pd.concat([
    test_pos_df,
    test_neg_df,
], axis=0, ignore_index=True)