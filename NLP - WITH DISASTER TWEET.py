#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import warnings
warnings.filterwarnings("ignore")


# In[1]:


#package importing
import pandas as pd


# In[2]:


#loading the dataset

df = pd.read_csv('train.csv')
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.shape


# #### Spliting data

# In[8]:


X= df.drop(['id','keyword','location','target'],axis=1)
y=df.pop('target')


# In[9]:


X.head()


# #### Preprocessing

# In[5]:


import nltk 


# In[6]:


nltk.download('stopwords')


# #### Converting to Lower case

# In[14]:


df["text_clean"] = df["text"].apply(lambda x: x.lower())


# #### Removing of URL form words

# In[16]:


def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

df["text_clean"] = df["text_clean"].apply(lambda x: remove_URL(x))


# #### remove link words and symbols

# In[17]:


def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

df["text_clean"] =df["text_clean"].apply(lambda x: remove_html(x))


# #### Remove emojis

# In[18]:


def remove_emojis(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df["text_clean"] = df["text_clean"].apply(lambda x: remove_emojis(x))


# In[19]:


def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))
df["text_clean"] = df["text_clean"].apply(lambda x: remove_punct(x))


# #### stopwords

# In[20]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

df['text'] = df['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))


# 
# #### tweettokenizer

# In[21]:


import nltk
from nltk import TweetTokenizer

tokenizer = TweetTokenizer()

df['tokens'] = [tokenizer.tokenize(item) for item in df.text]


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


X = vectorizer.fit_transform(df.text).toarray()
y = y


# #### Spliting the data

# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30,random_state=42)


# #### Creating the LogisticRegression Model

# In[25]:


from sklearn.linear_model import LogisticRegression

lor = LogisticRegression()
lor.fit(X_train,y_train)


# In[28]:


y_pred = lor.predict(X_test)
y_pred


# #### confusion matrix and classification report

# In[29]:


from sklearn.metrics import *


# In[30]:


cm=confusion_matrix(y_test, y_pred)
cm


# In[32]:


clr= classification_report(y_test, y_pred)
print(clr)


# #### Confusion matrix heatmap

# In[34]:


sns.heatmap(cm,annot = True,fmt='g')


# In[ ]:




