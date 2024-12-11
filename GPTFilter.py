import os
from pydoc_data.topics import topics

from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

import hybrid_queries_gioz
import gensim
import joblib
import numpy as np
import pandas as pd
import spacy
from openai import OpenAI
from src.gioz.hybrid_queries_gioz import hybrid_search_by_user_query_string_and_certainty
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
nltk.download('omw-1.4')

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_rows', 2025)
pd.set_option('display.max_columns', 1000)

os.environ['OPENAI_API_KEY'] = ""

client = OpenAI()

exquery = "Rinnovare Patente di guida"

def expand_query(query):
    words = query.split()
    expanded_query = set(words)
    for word in words:
        synsets = wordnet.synsets(word, lang = 'ita')
        print(dir(synsets))
        if synsets:
            for syn in synsets:
                ops = syn.lemma_names(lang = 'ita')
                for i in ops:
                    expanded_query.add(i)
    return expanded_query

print(expand_query(exquery))


og_search = hybrid_search_by_user_query_string_and_certainty(
        collection_name="AmministrazioneTrasparente",
        user_query_string=exquery,
        alpha=1,
        return_properties=["doc_source_uri", "chunk_text", "doc_atto_settore"],
        limit=10,
        prefiltering_clause=""
)

og_search_db = pd.DataFrame({"ID": og_search[1],
                            "Score": og_search[2],
                             "DocURL": og_search[3],
                             "Text": og_search[4]})

df = pd.read_pickle("/home/giovanni/Desktop/GitRepo/comune-padova/intelligent-virtual-assistant-for-public-administration/weaviate-starter-kit/src/gioz/postnmfDataFrame5k30t.pkl")
model = joblib.load("/home/giovanni/Desktop/GitRepo/comune-padova/intelligent-virtual-assistant-for-public-administration/weaviate-starter-kit/src/gioz/nmf_model5k30t.pkl")
vectorizer = joblib.load("/home/giovanni/Desktop/GitRepo/comune-padova/intelligent-virtual-assistant-for-public-administration/weaviate-starter-kit/src/gioz/nmf_vectorizer5k30t.pkl")
feature_names = vectorizer.get_feature_names_out()
print(feature_names[5875:5885])
print(df["ID"].head())
x = df[df["ID"].isin(og_search[1])].index

nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
top_keywords = []
for topic in model.components_:
    indexes = topic.argsort()[-10:]
    dec_idxs = indexes[::-1]
    top_keywords.append([feature_names[i] for i in dec_idxs])
print(top_keywords)
example = nlp("casa")
print(example.vector)
topic_vectors = []
top_keywords_withscore = []
for topic in model.components_:
    indexes = topic.argsort()[-10:]
    dec_idxs = indexes[::-1]
    top_keywords_withscore.append(([feature_names[i] for i in dec_idxs], [topic[j] for j in dec_idxs]))
print(top_keywords_withscore)
top = 0
for element in top_keywords_withscore:
    topicvec = [0] * 96
    tot_score = 0
    for i in range(10):
        vec = nlp(element[0][i]).vector
        norm_vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
        topicvec = topicvec + norm_vec * element[1][i]
        tot_score = tot_score + element[1][i]
    topic_vectors.append((top, topicvec / (10 * tot_score)))
    top = top + 1
print(topic_vectors)

#print(x[0] for x in df[:10])
def filterwithgpt(query, topn, topic_keywords, alpha):
    gooddocs = []
    for index, row in topn.iterrows():
        doc = row['topic_distribution']
        #id = row['ID']
        top3topics = (np.argsort(doc)[-7:][::1])
        print(top3topics)
        #print(topic_keywords[4])
        #doc_keywords = [(doc_keywords.extend(topic_keywords[i])) for i in top3topics]
        doc_keywords = topic_keywords[top3topics[0]] + topic_keywords[top3topics[1]] + topic_keywords[top3topics[2]] + topic_keywords[top3topics[3]] + topic_keywords[top3topics[4]] + topic_keywords[top3topics[5]] + topic_keywords[top3topics[6]]
        print(doc_keywords)
        prompt = "Given this first list of words: \"" + query + "\" and this second list of words: " + str(doc_keywords) + " tell me if it does exist a pair of words (one from the first list, one from the second) where the words are even slightly correlated with each other (on a scale of 1 to 100, more than "+ str(alpha) + "). Don't look at the theme of the words to evaluate, but just at the individual pairs of words. The first word of your answer has to be Yes or No, then you can explain further"
        print(prompt)

        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "user", "content": prompt}],
            max_completion_tokens = 150,
            n = 1,
            stop = None,
            temperature = 0.7
        )
        print(response.choices[0].message)
        if response.choices[0].message.content[:3] == "Yes":
            gooddocs.append(row)
            print("Document added")
    gooddata = pd.DataFrame(gooddocs)
    return gooddata





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

cleanquery = clean_text(exquery)
print(cleanquery)
print(expand_query(cleanquery))



print("cleanquery")
#query = list(sent_to_words(cleanquery))


def lemmatization(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)


lemquery = lemmatization(cleanquery)
query = str(expand_query(cleanquery))
print(query)
filtereddata = filterwithgpt(query, df.iloc[list(x)], [x[:3] for x in top_keywords], 20)

print("Lemmatized query:")
print(query)
print("splittedquery:")
print(query.split())
print(len(query.split()))
queryvec = [0] * 96
for word in query.split():
    vec = nlp(word).vector
    norm_vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
    queryvec = queryvec + norm_vec
queryvec = queryvec / len(query.split())
topic_query_similarity = []
print(queryvec, type(queryvec), len(queryvec))

for i in range(len(model.components_)):
    topic_query_similarity.append(np.dot(queryvec, topic_vectors[i][1]) / (np.linalg.norm(queryvec) * np.linalg.norm(topic_vectors[i][1])))
print("Topic_Query_SIM")
print(topic_query_similarity)
filtereddata['query_simil'] = None
for index, row in filtereddata.iterrows():
    filtereddata.at[index, 'query_simil'] = np.dot(row['topic_distribution'], topic_query_similarity) / (np.linalg.norm(row['topic_distribution']) * np.linalg.norm(topic_query_similarity))
print(filtereddata['query_simil'])

#print(filtereddata.sort_values(by = "query_simil", ascending = False, inplace = False))
iva_results = og_search_db.sort_values(by = "Score", ascending = False, inplace = False)
print(iva_results)
og_search_db['query_similarity'] = 0
for index, row in og_search_db.iterrows():
    for index2, row2 in filtereddata.iterrows():
        if row['ID'] == row2['ID']:
            og_search_db.at[index, 'query_similarity'] = row2['query_simil']
            break
print(og_search_db)
#og_search_db["Text"] = og_search_db["Text"].apply(clean_text)
texts = list(enumerate(og_search_db.Text.values.tolist()))
print(texts)
prompt2 = "Given this list of 3 enumerated italian texts in the form (index, text): "+ str(texts[:3]) + " and this query in italian: \"" + exquery + "\", assign to all the 4 texts a relevance score representing its relevance to the query (0 = Not relevant at all, 1 = Very little relevant, 2 = Partially relevant, 3 = Relevant). Start your answer with a list of the 3 relevance scores of the corresponding texts (example: 0,3,2)"
prompt3 = "Given this list of 3 enumerated italian texts in the form (index, text): "+ str(texts[3:6]) + " and this query in italian: \"" + exquery + "\", assign to all the 4 texts a relevance score representing its relevance to the query (0 = Not relevant at all, 1 = Very little relevant, 2 = Partially relevant, 3 = Relevant). Start your answer with a list of the 3 relevance scores of the corresponding texts (example: 0,3,2)"
prompt4 = "Given this list of 4 enumerated italian texts in the form (index, text): "+ str(texts[6:]) + " and this query in italian: \"" + exquery + "\", assign to all the 4 texts a relevance score representing its relevance to the query (0 = Not relevant at all, 1 = Very little relevant, 2 = Partially relevant, 3 = Relevant). Start your answer with a list of the 4 relevance scores of the corresponding texts (example: 0,1,3,2)"

print(prompt2)
system = "You are an italian man, that has to judge the relevance of text documents to a user query"
answer1 = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "user", "content": prompt2}],
            max_completion_tokens = 250,
            n = 1,
            stop = None,
            temperature = 0
        )
print(answer1.choices[0].message.content.splitlines()[0])
answer2 = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "user", "content": prompt3}],
            max_completion_tokens = 250,
            n = 1,
            stop = None,
            temperature = 0
        )
print(answer2.choices[0].message.content.splitlines()[0])
answer3 = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "user", "content": prompt4}],
            max_completion_tokens = 50,
            n = 1,
            stop = None,
            temperature = 0
        )
print(answer3.choices[0].message.content.splitlines()[0])

answer_text = answer1.choices[0].message.content.splitlines()[0] + "," + answer2.choices[0].message.content.splitlines()[0] + "," + answer3.choices[0].message.content.splitlines()[0]
print(answer_text)
splitted_answer = answer_text.split(',')
print(splitted_answer)
gpt_rank = [int(num) for num in splitted_answer]
print(gpt_rank)

IVA_relevance = np.array([[float(x) for x in og_search_db.Score.tolist()]])
print(IVA_relevance)
true_relevance = np.array([gpt_rank])
print(true_relevance)
#og_search_db["My_rank"] = og_search_db["Gpt_rank"].iloc[og_search_db['query_similarity'].sort_values(ascending = False).index].values
My_relevance = np.array([og_search_db.query_similarity.tolist()])
print(My_relevance)
my_ndcg = ndcg_score(true_relevance, My_relevance)
IVA_ndcg = ndcg_score(true_relevance, IVA_relevance)

print("MY NDCG: ", my_ndcg)
print("IVA NDCG: ", IVA_ndcg)