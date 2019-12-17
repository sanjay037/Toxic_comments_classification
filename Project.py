import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import tqdm
from tqdm import trange
import sys
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import pickle



def save_to_pickle(filename,model):
    pkl_file = filename
    pickle.dump(model,open(filename,'wb'))


# loading the training data and testing data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# storing the names of all column names except comment_text from training data
columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# initializing lemmatization
lemmatizer = WordNetLemmatizer()
# initializing Stemming
ps = PorterStemmer()
# initializing logistic regression with hyperparameters tuning
# log_reg = LogisticRegression(solver = "liblinear",C=5,random_state = 4)
# initializing random forest classifier
# rf = RandomForestClassifier(n_estimators=100)
# declaring stop words into a list stop_words
stop_words = set(stopwords.words('english'))
# removing no and not from stop words
stop_words.remove('no')
stop_words.remove('not')

# checking length of sentence of train data
plt.hist(train_data['comment_text'].apply(lambda x: len(str(x))))
plt.ylabel('no.of words')
plt.xlabel('length of word')
plt.show()

sns.countplot(train_data['toxic'],label="toxic")
plt.show()

sns.countplot(train_data['severe_toxic'],label="severe_toxic")
plt.show()

sns.countplot(train_data['obscene'],label="obscene")
plt.show()

sns.countplot(train_data['insult'],label="insult")
plt.show()

sns.countplot(train_data['identity_hate'],label="identity_hate")
plt.show()

sns.countplot(train_data['threat'],label="threat")
plt.show()


data_distribution = train_data[columns].sum()\
                            .to_frame()\
                            .rename(columns={0: 'count'})\
                            .sort_values('count')

data_distribution.plot.pie(y='count',
                                      title='Label distribution',
                                      figsize=(5, 5))\
                            .legend(loc='center left', bbox_to_anchor=(1.3, 0.5))

data_distribution.sort_values('count', ascending=False)

# plotting word cloud
word_counter = {}

def clean_text(text):
    text = re.sub('[{}]'.format(string.punctuation), ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in (stop_words)])

for categ in columns:
    d = Counter()
    train_data[train_data[categ] == 1]['comment_text'].apply(lambda t: d.update(clean_text(t).split()))
    word_counter[categ] = pd.DataFrame.from_dict(d, orient='index')\
                                        .rename(columns={0: 'count'})\
                                        .sort_values('count', ascending=False)

for w in word_counter:
    wc = word_counter[w]

    wordcloud = WordCloud(
          background_color='black',
          max_words=200,
          max_font_size=100,
          random_state=4561
         ).generate_from_frequencies(wc.to_dict()['count'])

    fig = plt.figure(figsize=(12, 8))
    plt.title(w)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# predefined set to change in the data
    defined_set = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(defined_set):
    contraction_re = re.compile('(%s)' % '|'.join(defined_set.keys()))
    return defined_set, contraction_re

contractions, contractions_re = _get_contractions(defined_set)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# preprocessinng the train data
sys.setrecursionlimit(5000)
array = [None]*len(train_data['comment_text'])
for i in trange(len(train_data['comment_text'])):
# replacing abbreviations with original words
    word_token = replace_contractions(train_data['comment_text'][i])
    word_token = word_token.replace("$","s")
    word_token = word_token.replace("f**k","fuck")
    word_token = word_token.replace("f*ck","fuck")
    word_token = word_token.replace("fck","fuck")
    word_token = word_token.replace("f@ck","fuck")
# removing links
    word_token = re.sub(r'http\S+', '', word_token)
# removing all characters except alphabets
    word_token = re.sub('[^A-Za-z]+', ' ', word_token)
# splinting sentence into words
    word_token = word_tokenize(word_token)
# converting phoneme into corresponding alphabet
    word_token = [unicodedata.normalize('NFKD',words).encode('ascii', 'ignore').decode('utf-8', 'ignore') for words in word_token]
# removing stop words and words of length of 1
    word_token = [words for words in word_token if not words in stop_words and len(words)>1]
# applying stemmatizer and lemmatizer and joining them into a sentence
    array[i] = ' '.join(ps.stem(lemmatizer.lemmatize(words)) for words in word_token)


# preprocessinng the test data
test_array = [None]*len(test_data['comment_text'])
for i in trange(len(test_data['comment_text'])):
# replacing abbreviations with original words
    word_token = replace_contractions(test_data['comment_text'][i])
    word_token = word_token.replace("$","s")
    word_token = word_token.replace("f**k","fuck")
    word_token = word_token.replace("f*ck","fuck")
    word_token = word_token.replace("fck","fuck")
    word_token = word_token.replace("f@ck","fuck")
# removing links
    word_token = re.sub(r'http\S+', '', word_token)
# removing all characters except alphabets
    word_token = re.sub('[^A-Za-z]+', ' ', word_token)
# splinting sentence into words
    word_token = word_tokenize(word_token)
# converting phoneme into corresponding alphabet
    word_token = [unicodedata.normalize('NFKD',words).encode('ascii', 'ignore').decode('utf-8', 'ignore') for words in word_token]
# removing stop words and words of length of 1
    word_token = [words for words in word_token if not words in stop_words and len(words)>1]
# applying stemmatizer and lemmatizer and joining them into a sentence
    test_array[i] = ' '.join(ps.stem(lemmatizer.lemmatize(words)) for words in word_token)


# inintializing TfidfVectorizer
tfidf = TfidfVectorizer()
# converting train data after pre-processing into a matrix using tfidf
vector = tfidf.fit_transform(array)
# converting test data after pre-processing into a matrix using tfidf
test_vector = tfidf.transform(test_array)


# save_to_pickle("Log_model.sav",log_reg)
# save_to_pickle("Rf_model.sav",rf)
load_rf = pickle.load(open("Log_model.sav","rb"))
load_log = pickle.load(open("Rf_model.sav","rb"))


# predicting using logistic regression
prediction_log = np.zeros((len(test_data), len(columns)))
for i, j in enumerate(columns):
    log_model = load_log.fit(vector,train_data[j])
    prediction_log[:,i] = log_model.predict_proba(test_vector)[:,1]

# predicting using random forest
prediction_rf = np.zeros((len(test_data), len(columns)))
for i, j in enumerate(columns):
    rf_model = load_rf.fit(vector,train_data[j])
    prediction_rf[:,i] = rf_model.predict_proba(test_vector)[:,1]


# final prediction is based on 0.8 times logistic prediction + 0.2 random forest prediction
prediction = np.zeros((len(test_data), len(columns)))
for i in range(len(test_data)):
    for j in range(len(columns)):
        prediction[i,j] = (0.8*prediction_log[i,j]+0.2*prediction_rf[i,j])


submission = pd.concat([pd.DataFrame({'id': test_data["id"]}), pd.DataFrame(prediction, columns = columns)], axis=1)
submission.to_csv('final_submission.csv', index=False)
