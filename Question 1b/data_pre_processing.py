from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
import time
from multiprocessing import Pool
import pandas as pd
import numpy as np


def pre_process(data_frame):
    # Remove blank rows if any.
    data_frame['Data'].dropna(inplace=True)

    # Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    data_frame['Clean_Data'] = data_frame['Data'].str.lower()

    # Tokenization: each entry will be broken into set of words
    # data_frame['Clean_Data'] = [word_tokenize(entry) for entry in data_frame['Clean_Data']]

    start = time.time()
    data_frame = parallelize_dataframe(data_frame, tkn)
    end = time.time() - start
    print(end)

    start = time.time()
    data_frame = parallelize_dataframe(data_frame, lemm)
    end = time.time() - start
    print("Total time on mc lemm:", end)
   
    return data_frame

def addToAvg(currentAvg, currentCount, value):
    return (currentAvg * currentCount + value) / (currentCount + 1)


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 10)
    pool = Pool(4)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def tkn(df):
    df['Clean_Data'] = df['Data'].apply(word_tokenize)
    return df

def lemm(df):

    # for index,entry in enumerate(df['Clean_Data']):
    #     # Declaring Empty List to store the words that follow the rules for this step
    #     Final_words = []

    #     # Initializing WordNetLemmatizer()
    #     word_Lemmatized = WordNetLemmatizer()


    #     start = time.time()
    #     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    #     for word, tag in pos_tag(entry):
    #         # print(word)
    #         # time.sleep(1)
    #         # Below condition is to check for Stop words and consider only alphabets
    #         if word not in stopwords.words('english') and word.isalpha():
    #             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
    #             Final_words.append(word_Final)

        
    #     end = time.time() - start

    #     avg = addToAvg(avg, cnt, end)
    #     cnt = cnt + 1
      

    #     # The final processed set of words for each iteration will be stored in 'text_final'
    #     df.loc[index,'Clean_Data'] = ' '.join(Final_words)
    df["Clean_Data"] = df["Clean_Data"].apply(inner_lem)
    
    return df

def inner_lem(entry):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    avg = 0
    cnt = 0
    
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []

    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()


    start = time.time()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # print(word)
        # time.sleep(1)
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)

    
    end = time.time() - start

    avg = addToAvg(avg, cnt, end)
    cnt = cnt + 1
    # print(avg)  
    return ' '.join(Final_words)