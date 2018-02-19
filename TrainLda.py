
# coding: utf-8

# In[1]:


#! /usr/bin/env python
import sys , os, re, string
import nltk
import jieba
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from collections import defaultdict
# import matplotlib.pyplot as plt
# %matplotlib inline


# In[20]:


stop_words_cn = []
with open("ch_stopwords.txt") as f_cn_stopword:
    for line in f_cn_stopword:
        stop_words_cn.append(line.strip("\n"))
    print(len(stop_words_cn))
#     for w in stop_words_cn:
#         print(w)


# In[21]:


# https://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
# path = "./enpod"

class Tokenizer:
    def if_stopword(word):
        stop_list = stopwords.words("english") +             [
                ",", ".", "!", "?", ";", ":", "\n", "\t",   \
                "(", ")", " ", "`", "'", "a", "b", "c",     \
                "’", "'m", "'s", "n't", "'ll", "'re",       \
                "'ve", "”", "“", "'d", "‘", "’", "–"
            ] + stop_words_cn
        if word.lower() in stop_list:
            return True
        else:
            return False

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        
        try:
            import re
            num_reg = re.compile("\\d+,*:*'*\\d+")
            if num_reg.match(s):
                return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
    
        return False
    
    def is_ascii(s):
        try:
            return all(ord(c) < 256 for c in s)
        except TypeError:
            return False
        
    # https://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokens(text):
        return (nltk.word_tokenize(text))
        
    def stems(text):
        porter_stemmer = PorterStemmer()
        tokens1 = nltk.word_tokenize(text)
        tokens = [x.lower() for x in tokens1 if not re.fullmatch('[' + string.punctuation + ']+', x) and not Tokenizer.if_stopword(x) and not Tokenizer.is_number(x)]
        stems = Tokenizer.stem_tokens(tokens, porter_stemmer)
        return stems


# In[4]:


class Documents(object):
    def __init__(self, *args):
        super(Documents, self).__init__(*args)
        
    def get_doc_name_list(self):
        # return self.__doc_file_name_list
        raise NotImplementedError()

    def get_doc_stems_list(self):
        raise NotImplementedError()
        # return self.__doc_stems_list
    def get_word_dict(self):
        raise NotImplementedError()


# ## 新概念英語

# In[5]:


class NceDOC(Documents):
    __doc_stems_list = []
    __doc_name_list = []
    __doc_word_dict = defaultdict(lambda x: 0)
    
    def __init__(self, path):
        with open(path, "r") as file:
            for i, line in enumerate(file):
                
                stem_list = Tokenizer.stems(line.strip())
                word_list = Tokenizer.tokens(line.strip())

                self.__doc_stems_list.append(stem_list)
                self.__doc_name_list.append("Doc%02d"%(i))
                self.__doc_word_dict = CountVectorizer()
                

    def get_doc_name_list(self):
        return self.__doc_name_list
        # raise NotImplementedError()

    def get_doc_stems_list(self):
        return self.__doc_stems_list

    def get_word_dict(self):
        return self.__doc_word_dict
        


# ## EnglishPod 

# In[6]:


class EpDOC(Documents):
    '''
    English Pod 
    '''
    # self.__path = ""
    __doc_stems_list = []
    __doc_file_name_list = []

    def __init__(self, path):
        # super(Documents, self).__init__(*args))
        self.__doc_stems_list = []
        self.__path = path
        for file_name in os.listdir(self.__path):
            word_list = []
            if not file_name.endswith("lrc"):
                continue
            file_path = os.path.join(path, file_name)
            print(file_name, end=': ')
            with open(file_path, "r") as file:
                for i, line in enumerate(file):
                    if i ==0 :
                        continue
                    stem_list = Tokenizer.stems(line.strip())
                    # stem_tokens(tokens, porter_stemmer)
                    for stem in stem_list:
                        # if not if_stopword(word):
                        print(stem, end=", ")
                        word_list.append(stem)
                        
                print("")
            self.__doc_stems_list.append(word_list)
            self.__doc_file_name_list.append(file_name)

    def get_doc_name_list(self):
        return self.__doc_file_name_list

    def get_doc_stems_list(self):
        return self.__doc_stems_list
        


# In[27]:


jieba.suggest_freq('云计算', True)
jieba.suggest_freq('区块链', True)
jieba.suggest_freq("物联网", True)
jieba.suggest_freq("大数据", True)
jieba.suggest_freq("智能终端", True)
jieba.suggest_freq("长达", True)
jieba.suggest_freq("低带宽", True)
jieba.suggest_freq("莫伦科夫", True)
jieba.suggest_freq("乐视网", True)
jieba.suggest_freq("饿了吗", True)
jieba.suggest_freq("于亚辉", True)
jieba.suggest_freq("配送员", True)


# In[8]:


class Xinhua(Documents):
    """
    新华网
    """
    __doc_stems_list = []
    __doc_file_name_list = []
    
    def __init__(self, path):
        reg_percentage = re.compile("[0-9.]+%")
#         path = "/Users/aron/xinhua/txt_xinhua"
        doc_array=[]
        for file_name in os.listdir(path):
            word_list = []
            if not file_name.endswith("txt"):
                continue
        #     print(os.path.join(path, file_name))

            doc = []
            with open(os.path.join(path, file_name), "r") as txt:
                for i, line in enumerate(txt):
                    if i == 0 or line.strip() == "" :
                        continue
                    ll = jieba.cut(line.strip())

                    for w in ll:
                        if Tokenizer.is_ascii(w) or Tokenizer.if_stopword(w) or Tokenizer.is_number(w) or w ==" ":
                            continue
                        m = reg_percentage.match(w)
                        if m:
                            continue
                        if len(w) == 1 and ord(w) < 256:
                            continue
                        
                        if len(w) == 1 and ord("Ａ") <= ord(w) <= ord("ｚ"):
                            continue
                        
                        word_list.append(w)
            self.__doc_stems_list.append(word_list)
            self.__doc_file_name_list.append(file_name)

    def get_doc_name_list(self):
        return self.__doc_file_name_list

    def get_doc_stems_list(self):
        return self.__doc_stems_list


# In[22]:


class LDA(object):
    # __docs = None
    def __init__(self, documents, n_topic_num):
        self.__docs = documents
        # super(LDA, self).__init__(doc_stems_list))
        vectorizer = CountVectorizer(tokenizer=lambda a: a, analyzer=lambda a: a)
        # vectorizer.fit(doc for doc in doc_list)
        vector_list = vectorizer.fit_transform(self.__docs.get_doc_stems_list())

        self.__feat_names = vectorizer.get_feature_names()
        # pd_vectorlist = pd.DataFrame(vector_list)#, columns=feat_names)
        # pd_vectorlist.to_csv("vectorlist.csv")
        self.__n_topic_num = n_topic_num
        self.__model = LatentDirichletAllocation(n_components=n_topic_num,
                                            doc_topic_prior=0.02,
                                            topic_word_prior=0.05,
                                            max_iter=50,
                                            learning_method='online',
                                            learning_offset=50.,
                                            random_state=0)
        self.__model.fit(vector_list)
        self.__doc_topic_dist = self.__model.transform(vector_list)
        
    def feat_names(self):
        return self.__feat_names 

    def word_distribution(self):
        normalize_components = np.array(self.__model.components_ / np.matrix (self.__model.components_.sum(axis=1)).T)
        return normalize_components

    def word_distribution_csv(self, path = "nce_words_dist.csv"):
        normalize_components = np.array(self.__model.components_ / np.matrix (self.__model.components_.sum(axis=1)).T)
        wordprob = pd.DataFrame(normalize_components, index =["#%02d"%(i) for i in range(self.__n_topic_num)])
        feat_name_dict = dict(zip(range(len(self.__feat_names )), self.__feat_names ))
        wordprob.rename(columns = feat_name_dict, inplace=True)
        w = wordprob.transpose()
        w.to_csv(path)
        return w

    def topic_distribution(self):
        return self.__doc_topic_dist

    def topic_distribution_csv(self, path="nce_topics_dist.csv"):
        pd_doc_topic_dist = pd.DataFrame(self.__doc_topic_dist, index = self.__docs.get_doc_name_list())

        topic_name_dict = dict(zip(range(self.__n_topic_num),["#%02d"%(i) for i in range(self.__n_topic_num)]))

        pd_doc_topic_dist.rename(columns = topic_name_dict, inplace=True)
        # pd_doc_topic_dist.to_csv("enpod_topics_data.csv")
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)

        title, ext = os.path.splitext(base_name)
        filename_t = title + "_T" + "." + ext
        pd_doc_topic_dist.transpose().to_csv(os.path.join(dir_name, filename_t))
        pd_doc_topic_dist.to_csv(path)
        return pd_doc_topic_dist


# In[10]:


# docs = EpDOC( "./enpod")
# docs = NceDOC("../nce/all.txt")
path = "/Users/aron/xinhua/txt_xinhua"
docs = Xinhua(path)
# print(docs.feat_names())


# In[11]:


# for doc_name in docs.get_doc_name_list():
    # print(doc_name)
lda = LDA (docs, 20)


# In[12]:


print(len(lda.feat_names()))


# In[13]:


topics_dist = lda.topic_distribution_csv(path="xinhua_topics_dist.csv")
word_dist = lda.word_distribution_csv(path="xinhua_word_distribution.csv")


# In[14]:


word_dist.to_csv("xinhua_word_distribution.csv", encoding="utf_8_sig")


# In[17]:


print(os.path.dirname("/Users/aron/dev/lda/notebook/s.s"))
print(os.path.basename("/Users/aron/dev/lda/notebook/s.s"))


# In[28]:


l = jieba.cut("点餐体验")
print([w for w in l])

