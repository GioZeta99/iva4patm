import spacy
import gensim



import numpy as np



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import random_chunks_from_collection_gioz
from src.gioz.random_chunks_from_collection_gioz import get_random_contents

data = get_random_contents()

# returns JSON object as
# a dictionary

newdf = pd.DataFrame({
    "ID": [x[0] for x in data],
    "ChunkText": [x[1] for x in data],
    "DocURL": [x[2] for x in data]
})

# Iterating through the json
# list



pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_rows', 2025)
pd.set_option('display.max_columns', 1000)
'''newdf = pd.DataFrame(columns = ['DocUrl', 'ChunkText'])
docid = 1
print(df.shape[0])
for chunk_id in range(df.shape[0]):
    newdf.loc[chunk_id] = [df.iloc[chunk_id].results['doc_source_uri'], df.iloc[chunk_id].results['chunk_text']]

print(newdf)
DocId = []
DocId.append(1)
for chunk_id in range(1, df.shape[0]):
    if newdf.iloc[chunk_id]['DocUrl'] == newdf.loc[chunk_id - 1]['DocUrl']:
        DocId.append(DocId[-1])
    else:
        DocId.append(DocId[-1] + 1)


print(DocId)
newdf['DocId'] = DocId
'''
dimdb = print(newdf.shape[0])

import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
nltk.download('stopwords')
nltk.download('wordnet')


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]|@,\';−–\-’]')
BAD_SYMBOLS_RE = re.compile(r'[0123456789#+*_●°♦“”.!:�=•<>’?×\"…&%$€\\]')
A_CAPO = re.compile('[\n]')
SINGLEWORDS = re.compile('\\b[A-Za-z] \\b|\\b [A-Za-z]\\b')
STOPWORDS = set(stopwords.words('italian'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.replace("logo_col.gif", '')
    text = A_CAPO.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = text.replace("pagina", '')
    #text = text.replace("x", '')
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace("\\", ' ')

    #text = text.replace(" n ", ' ')
    #text = text.replace(" r ", ' ')
    #text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    text = SINGLEWORDS.sub('', text)
    return text

print(clean_text("l'alluvione ha- colpito x cose a mantova () - lol 3 \n sdf"))
def cut_low_text_length(array, k, m):
    new_index = []
    for i in range(m):
        if len(array[i]) < k:
            new_index.append(i)
    #print(new_index)
    return new_index



newdf['ChunkText'] = newdf['ChunkText'].apply(clean_text)

print(type(newdf['ChunkText']))
#small_chunks_id = cut_low_text_length(newdf['ChunkText'], 80, newdf.shape[0])
#newdf = newdf.drop(index = small_chunks_id)
#newdf = newdf.reset_index()

import gensim

from gensim.utils import simple_preprocess

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data = newdf.ChunkText.values.tolist()



data_words = list(sent_to_words(data))

query = "Il sole splende sulla citta"
print(query)
query = list(sent_to_words(query))
print(data_words[:1][0][:30])

from sklearn.decomposition import NMF
from gensim.utils import simple_preprocess


def display_topics(H, feature_names, num_top_words):
    for topic_idx, topic in enumerate(H):
        print(f"Topic {topic_idx + 1}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1 : -1]]))
        print()


vectorizer = TfidfVectorizer(max_features = 5000, stop_words = None)
tfidf_matrix = vectorizer.fit_transform(data)
nmf_model = NMF(n_components = 25, random_state = 41)
W = nmf_model.fit_transform(tfidf_matrix)
H  = nmf_model.components_


feature_names = vectorizer.get_feature_names_out()
display_topics(H, feature_names, 5)

import joblib

joblib.dump(nmf_model, 'nmf_model5k30t.pkl')
#joblib.dump(W, 'nmf_W100k.pkl')
joblib.dump(vectorizer, 'nmf_vectorizer5k30t.pkl')

topic_distributions = nmf_model.transform(tfidf_matrix)
newdf['topic_distribution'] = topic_distributions.tolist()
print(newdf.head())

newdf.to_pickle("postnmfDataFrame5k30t.pkl")

'''000013b9-c390-403b-80e8-4f74b84405d1   
1      1  00002d0f-883b-4007-bec2-9992743c6b93   
2      2  0000efa9-5ce8-4eb8-95f9-c5cc9188f5d8   
3      3  0001212c-0ebc-47b1-adeb-f1e6e1ae17a1   
4      4  00015d1e-df87-475e-b519-6b5a2903f337   '''
