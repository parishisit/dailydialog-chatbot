import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#read data
txt=open('dialogues_text.txt').read()
act=open('dialogues_act.txt').read()
emo=open('dialogues_emotion.txt').read()
topic=open('dialogues_topic.txt').read()

#splitting by lines
dialogs= txt.lower().split('\n')
acts= act.split('\n')
emos= emo.split('\n')
topics= topic.split('\n')

#assigning each emotion to its corresponding text
emos_=[]
for emo_line in emos[0:]:
  emo_line_= emo_line.split(' ')
  del emo_line_[-1]
  emos_.append(emo_line_)

#assigning each act to its corresponding text
acts_=[]
for act_line in acts[0:]:
  act_line_= act_line.split(' ')
  del act_line_[-1]
  acts_.append(act_line_)

#making vectors of our data
Q=[]
A=[]
emo=[]
topic=[]
act=[]
for i in range(0, len(dialogs)):
  lines= dialogs[i].split(' __eou__')
  del lines[-1]
  for j in range(0, len(lines)-1):
    Q.append(lines[j])
    A.append(lines[j+1])
    emo.append(emos_[i][j])
    topic.append(topics[i])
    act.append(acts_[i][j])

#tfidf vectorizer    
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(Q)
q_tfidf = tfidf_vectorizer.transform(Q)

#saving our vectors and the vectorizer
import pickle
with open("raw.pickle", "wb") as f:
  pickle.dump((Q, A, emo, topic, act, q_tfidf, tfidf_vectorizer), f)
