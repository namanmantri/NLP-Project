# importing required modules
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk import RegexpParser
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import codecs
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
import os
import re
from nltk.sem.relextract import extract_rels, rtuple
import seaborn as sn
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import ieer
from spacy import displacy
from nltk.sem import extract_rels,rtuple

# downloading required modules
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')

# reading 'Sherlock' book
Sherlock_book = open(r"Sherlock.txt",encoding='utf-8')
Sherlock_book=Sherlock_book.read()

#preprocessing book
Sherlock_book=re.sub(r"http\S+", "", Sherlock_book,flags=re.MULTILINE)#For http
Sherlock_book=re.sub(r' www\S+',' ', Sherlock_book) #For www.
Sherlock_book=re.sub(r'[^(a-zA-Z)\s]','',Sherlock_book) #For charcters other than alphabets
Sherlock_book=re.sub(r' (\n)',' ',Sherlock_book)#For removing sentnce gaps
Sherlock_book=re.sub(r' ','  ',Sherlock_book) #For removing extra space
Sherlock_book=re. sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b','',Sherlock_book)#For chapter names

# tokenizing on sherlock book
S_tokenized=word_tokenize(Sherlock_book)

#Creating Stopwords
stop_words = set(stopwords.words('english'))
stop_words_2=set(STOPWORDS)

#Removing Stopwords from Sherlock tokens
S_tokens_sw = [i for i in S_tokenized if not i in stop_words and not i in stop_words_2]
S_text_sw = " "
S_text_sw = S_text_sw.join(S_tokens_sw) 

#function to find sense of word using Lesk algorithm
def findCategories(tokens,tags,nouns,verbs):
    for word in tokens:
        if not lesk(tokens,word):
            continue
        if lesk(tokens, word).pos() == 'n': # if sense of the word is noun
            category = lesk(tokens, word).lexname()
            if category not in nouns.keys():
                nouns[category] = 1 # if a noun category is found for the first time
            else:
                nouns[category] += 1 # incrementing the count of noun category
        elif lesk(tokens, word).pos() == 'v':
            category = lesk(tokens, word).lexname()
            if category not in verbs.keys():
                verbs[category] = 1 # if a verb category is found for the first time
            else:
                verbs[category] += 1 # incrementing the count of verb category


# plotting graph
def plot(X,Y,xlabel,ylabel,title):
    fig = plt.figure(figsize = (15, 5))    
    plt.bar(X, Y, color ='maroon',width = 0.4) 
  
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.title(title) 
    plt.show()    

#POS Tagging
pos_tagged=nltk.pos_tag(S_tokens_sw)
#print(pos_tagged)

#creating plot for postags
tags = {}
for u,v in pos_tagged:
    if v not in tags.keys():
        tags[v] = 1
    else:
        tags[v] += 1

X = []
Y = []
for i in tags.keys():
    X.append(i)

for i in X:
    Y.append(tags[i])

# plotting relationship between tags and frequency
xlabel = 'tags'
ylabel = 'frequency'
title = 'Relationship between tags and frequency'
plot(X,Y,xlabel,ylabel,title)

# finding words belonging to noun and verb category              
nouns = {}
verbs = {}
# below function categories the word as noun/verb category
findCategories(S_tokens_sw,pos_tagged,nouns,verbs)

X = []
Y = []
for noun in nouns.keys():
    X.append(noun.split('.')[1][:4])
    Y.append(nouns[noun])


xlabel = 'noun categoraies'
ylabel = 'frequency'
title = 'Relationship between noun categories and their frequency'
#plotting Relationship between noun categories and their frequency
plot(X,Y,xlabel,ylabel,title)

X = []
Y = []
for verb in verbs.keys():
    X.append(verb.split('.')[1][:4])
    Y.append(verbs[verb])


xlabel = 'verb categories'
ylabel = 'frequency'
title = 'Relationship between verb categories and their frequency'
#plotting Relationship between verb categories and their frequency
plot(X,Y,xlabel,ylabel,title)
    
#funtion to perform named entity recognition                      
nlp = spacy.load('en_core_web_sm')
book_S_text=nlp(Sherlock_book)
S_entity=[' '] # stores the word(entity)
S_entity_type=[' ']#stores the type of entity(Person etc.)
[(S_entity.append(X.text),S_entity_type.append(X.label_))  for X in book_S_text.ents]

#displaying a paragraoh after named entity recognition
displacy.render(nlp(Sherlock_book[2000:4000]), jupyter=True, style='ent')

new = defaultdict(list)  # dict in which key is entity type and value is list of words of that entity
print(S_entity)
print(S_entity_type)
k=0
for u in S_entity_type:
    if u not in new:
        new[u] = [S_entity[k]]
    else:
        new[u].append([(S_entity[k])])
    k+=1


# entity depiction                                             
displacy.render(nlp(Sherlock_book[3000:3100]), jupyter=True, style='dep')

#checking accuracy using human labelling and prediced labels
def confusionMatrix(results,entities,entity):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    actual = 0

    if entity in entities.keys():
        total_pos_predicted = len(entities[entity])
    else:
        total_pos_predicted = 0

    total_neg_predicted = len(results)-total_pos_predicted

    for line in results:
        name,e = line[0].lower(),line[1]
        if e == entity:
            actual += 1
            if entity in entities.keys():
                if name in entities[entity]:
                    true_pos += 1
        
            
            
    false_pos = total_pos_predicted-true_pos
    false_neg = actual-true_pos
    true_neg = total_neg_predicted-false_neg

    recall = true_pos/(true_pos+false_neg)
    precision = true_pos/(true_pos+false_pos)
    #print(recall,precision)
    fscore = 2*recall*precision/(recall+precision) #calculating f score

    print('F-measue =',fscore)

    matrix = [[true_pos,false_pos],[false_neg,true_neg]]

    df_cm = pd.DataFrame(matrix, index = [i for i in ['Positive','Negative']],
                  columns = [i for i in ['Positive','Negative']])

    xlabel = 'Actual Values'
    ylabel = 'Predicted Values'
    title = 'Confusion Matrix of '+entity
    ax = plt.subplot()
    sn.heatmap(df_cm, annot=True,ax = ax) # creating heatmap
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()
    
    
results = []
file = open('label.txt','r') # human annotated text 
lines = file.readlines()
n = int(lines.pop(0))
for i in range(n):
    vals = lines.pop()
    vals = vals.split('=')
    line = []
    for j in vals:
        line.append(j.replace('\n','').replace('\t',''))
    results.append(tuple(line))

    
confusionMatrix(results,new,'PERSON') # calling confusion matrix function for PERSON   
confusionMatrix(results,new,'LOC') # calling confusion matrix function for LOC  
    
    
# depicting relationship between entities
sherlock_sample=Sherlock_book
sherlock_sentence = nltk.sent_tokenize(sherlock_sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sherlock_sentence]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]


OF = re.compile(r'.*\bof\b.*') # regular expression for of relation
IN = re.compile(r'.*\bin\b(?!\b.+ing)') # regular expression for in relation
print('PERSON-ORGANISATION Relationships:')
for i, sent in enumerate(tagged_sentences):
    sent = nltk.ne_chunk(sent) # ne_chunk method expects one tagged sentence
    rels = extract_rels('PER', 'ORG', sent, corpus='ace', pattern=OF, window=7) # extract_rels method expects one chunked sentence
    for rel in rels:
        print(rtuple(rel))

print()
print('PERSON-GPE Relationships:')
for i, sent in enumerate(tagged_sentences):
    sent = nltk.ne_chunk(sent) # ne_chunk method expects one tagged sentence
    rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=OF, window=10) # extract_rels method expects one chunked sentence
    for rel in rels:
        print(rtuple(rel))
    


















