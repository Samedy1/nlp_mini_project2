from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

# -------------------- Read Data --------------------
with open('data/positive-reviews.txt') as infile:
    positive_lst = infile.read().split('\n')
    
with open('data/negative-reviews.txt') as infile:
    negative_lst = infile.read().split('\n')
    
with open('data/positive-words.txt') as infile:
    positive_words = infile.read().split('\n')
    
with open('data/negative-words.txt', encoding='latin-1') as infile:
    negative_words = infile.read().split('\n')

# -------------------- Create a Dataframe --------------------
def create_df(review: list, sentiment=1):
    return pd.DataFrame({
    'review': review,
    'sentiment': [sentiment for x in review]
})

# -------------------- Preprocess --------------------
def preprocess(lst: list[str], sentiment=1):
    new_df = create_df(lst, sentiment=sentiment)
    new_df['cleaned_review'] = new_df['review'].apply(clean_data)
    new_df = create_extracted_features(new_df, from_feature='cleaned_review')
    return new_df

def clean_data(text: str, transform_only: bool = False) -> str:
    # sentence tokenization
    sentences = sent_tokenize(text)

    # lowercase
    sentences = [sentence.lower() for sentence in sentences]

    # word tokenization
    tokens = [word_tokenize(sentence) for sentence in sentences]

    # remove stopwords
    en_stopwords = stopwords.words('english')
    used_pronouns = ['i', 'me', 'my', 'you', 'your']
    for value in used_pronouns: 
        if(value in en_stopwords): 
            en_stopwords.remove(value)

    no_stopwords = []
    for lst in tokens:
        no_stopwords.append([token for token in lst if token not in en_stopwords])
    
    # remove puntuations, and numbers, keep !
    alphabet_pattern = re.compile(r'[a-z!]+')
    alphabet_tokens = []
    for lst in no_stopwords:
        alphabet_tokens.append([''.join(alphabet_pattern.findall(token)) for token in lst])
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for lst in alphabet_tokens:
        tagged_tokens = pos_tag(lst)
        lemmatized_tokens.append([lemmatizer.lemmatize(token) for token, pos in tagged_tokens])

    # remove empty string
    no_empty = []
    for lst in lemmatized_tokens:
        no_empty.append([token for token in lst if token != ''])

    cleaned_text = ''
    for lst in no_empty:
        cleaned_text = cleaned_text + ' '.join(lst)
        
    return cleaned_text

# -------------------- Feature Extraction --------------------
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

def get_no_existed(text):
    return 1 if 'no' in text.split(' ') else 0

def get_pronoun_count(text):
    pronouns = ['i', 'me', 'my', 'you', 'your']
    return len([word for word in text.split(' ') if word in pronouns])

def get_exclamation_mark_existed(text):
    return 1 if '!' in text.split(' ') else 0

def get_log_word_count(text: str):
    return np.log(len(text.split(' ')))

def create_extracted_features(df: pd.DataFrame, from_feature: str):
    df2 = df.copy()
    df2['x1'] = df2[from_feature].apply(get_count_positive_words)
    df2['x2'] = df2[from_feature].apply(get_count_negative_words)
    df2['x3'] = df2[from_feature].apply(get_no_existed)
    df2['x4'] = df2[from_feature].apply(get_pronoun_count)
    df2['x5'] = df2[from_feature].apply(get_exclamation_mark_existed)
    df2['x6'] = df2[from_feature].apply(get_log_word_count)
    return df2

# -------------------- Split Data --------------------
# size of training data in positive df
training_pos_size = int(len(positive_lst) * 0.8)

# size of training data in negative df
training_neg_size = int(len(positive_lst) * 0.8)

# -------------------- Convert the List into Preprocessed DataFrame --------------------
# positive reviews
training_pos_df = preprocess(positive_lst[:training_pos_size])
test_pos_df = preprocess(positive_lst[training_pos_size:])

# negative reviews
training_neg_df = preprocess(negative_lst[:training_neg_size], sentiment=0)
test_neg_df = preprocess(negative_lst[training_neg_size:], sentiment=0)

# -------------------- Combine --------------------
training_df = pd.concat([
    training_pos_df,
    training_neg_df,
], axis=0, ignore_index=True)

test_df = pd.concat([
    test_pos_df,
    test_neg_df,
], axis=0, ignore_index=True)

# -------------------- Selected Input Features --------------------
input_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']