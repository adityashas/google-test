import re 
import nltk 
import torch 
import json 
import pickle, ssl 
from tqdm import tqdm
import pandas as pd
import numpy as np 
import itertools,validators
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from google.cloud import storage

#ssl._create_default_https_context = ssl._create_unverified_context

mymodel = torch.load('models/cygnus.pt')
model1 = pickle.load(open('models/tmodel5.pkl', 'rb'))
le1 = pickle.load(open('models/tlabel5.pkl', 'rb'))


class TaxonomyPredictor:
    def __init__(self, data):
        self.data = data         

    def max_sum_sim(self):
        distances = cosine_similarity(self.doc_embedding, self.candidate_embeddings) 
        distances_candidates = cosine_similarity(self.candidate_embeddings,self.candidate_embeddings) 
        words_idx = list(distances.argsort()[0][-self.nr_candidates:]) 
        words_vals = [self.candidates[index] for index in words_idx] 
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]  
        min_sim = np.inf 
        candidate = None 
        for combination in itertools.combinations(range(len(words_idx)), self.top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j]) 
            if sim < min_sim: 
                candidate = combination 
                min_sim = sim     
        scores = sorted(distances[0], reverse=True)
        return [(words_vals[idx],scores[idx]) for idx in candidate]

    def content_fetcher(self):
        url = self.data
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req, timeout=20).read()
        soup = BeautifulSoup(html, features="lxml")
        text = ' '
        for data in soup.find_all("p"):
            if 'JavaScript ' not in data.get_text() and 'We use cookies' not in data.get_text() :
                text += ' ' + data.get_text()
        return text 

    def predict1(self):
        #text = self.vkeys
        text = self.doc
        vector = mymodel.encode([text])
        preds = model1.predict_proba(vector)[0]
        preds_list = [pred for pred in preds]
        l1 = {}
        names = list(le1.classes_)
        for n, pred in zip(names, preds_list):
           l1[n] = pred
        p = sorted(l1.items(), key=lambda item: item[1], reverse=True) 
        k1={k: v for k, v in p} 
        k=list(k1.items())
        l1 = []
        for n in range(2):
            l1.append(k[n][0])
        return l1

    def taxonomy_predictor(self):
        n_gram_range1 = (1, 1)  
        n_gram_range2 = (3, 3)  
        stop_words = "english"
        self.top_n=25
        self.nr_candidates=25
        if validators.url(self.data):
            text = self.content_fetcher()
        else:
            text = self.data
        data1=text.lower() 
        doc=re.sub('[^a-z]+',' ', data1) 
        self.doc = re.sub(r'\b\w{1,2}\b', '', doc)
        count = CountVectorizer(ngram_range=n_gram_range1, stop_words=stop_words).fit([self.doc]) 
        self.candidates = count.get_feature_names() 
        self.doc_embedding = mymodel.encode([self.doc]) 
        self.candidate_embeddings = mymodel.encode(self.candidates) 
        keys = self.max_sum_sim() 
        nkey = []
        for k in keys:
            nkey.append(k[0])
        keywords =  ' '.join(nkey)
        count2= CountVectorizer(ngram_range=n_gram_range2, stop_words=stop_words).fit([keywords]) 
        self.candidates = count2.get_feature_names() 
        self.doc_embedding = mymodel.encode([keywords]) 
        self.candidate_embeddings = mymodel.encode(self.candidates) 
        self.top_n=len(self.candidates)
        self.nr_candidates=len(self.candidates)
        mss = self.max_sum_sim() 
        longkeys = []
        for m in mss:
            longkeys.append(m[0])
        self.vkeys = ' '.join(longkeys)
        print(longkeys)
        name1 = self.predict1()
        rdict = {'description':doc,'keywords':keys,'long keywords':mss, 'taxonomy':name1}
        return rdict

#tp = TaxonomyPredictor('https://www.activestate.com/blog/phishing-url-detection-with-python-and-ml/')
#preds = tp.taxonomy_predictor()
#print(preds)