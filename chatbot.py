import os
import pickle
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class chatbot():
    
    def __init__(self):
        self.default=["Sorry, I do not understand.", "Umm, would you please clarify?", "I do not get what you mean!", "That is beyond my context!"]
        self.bye=['bye','quit','goodbye','finish','end']
        with open("raw1.pickle", "rb") as f:
            self.q, self.a ,self.emo ,self.topic ,self.act, self.q_tfidf, self.tfidf_vectorizer= pickle.load(f)
            
    def match(self, query, threshold=0.6):
        query=query.lower()
        query_tfidf= self.tfidf_vectorizer.transform([query])
        vals = cosine_similarity(query_tfidf, self.q_tfidf)[0]
        idxs= np.where(vals>=threshold)[0]
        return idxs
    
    def response(self, query):
        idxs=self.match(query)
        if len(idxs)==0:
            return self.default[random.randrange(0,len(self.default))]
        else:
            idx=idxs[random.randrange(0,len(idxs))]
            return self.a[idx]
        
    def chat(self):
        print("Hello! Let's talk")
        while True:
            query=input().lower()
            if query not in self.bye:
                resp= self.response(query)
                print(resp)
            else:
                print("It was nice talking to you, bbye!")
                break


def main():
    cb=chatbot()
    cb.chat()

if __name__=="__main__":
    main()

    
