import nltk
from nltk import FreqDist 
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt 
import pandas as pd
from nltk.tokenize import word_tokenize
import seaborn as sns 


#downloading Required files of nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


#Reading Sherlock Book
Sherlock_book = open(r"Sherlock.txt",encoding='utf-8')
Sherlock_book=Sherlock_book.read()

#Cleaning The sherlock book
Sherlock_book=re.sub(r"http\S+", "", Sherlock_book,flags=re.MULTILINE)#For http
Sherlock_book=re.sub(r' www\S+',' ', Sherlock_book) #For www.
Sherlock_book=re.sub(r'[^(a-zA-Z)\s]','',Sherlock_book) #For charcters other than alphabets
Sherlock_book=re.sub(r' (\n)',' ',Sherlock_book)#For removing sentnce gaps
Sherlock_book=re.sub(r' ','  ',Sherlock_book) #For removing extra space
Sherlock_book=re. sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b','',Sherlock_book)#For chapter names

#Tokenizing Sherlock Book
S_tokenized=word_tokenize(Sherlock_book)

#Finding frequency distrubution of token
S_freq=FreqDist(S_tokenized)

#Using seaborn to plot frequency 
sns.set(style='whitegrid') 

#Converting tokenized values into dataframe
df=pd.DataFrame(S_tokenized)

#Plotting the tokens
sns.countplot(x=df[0], order=df[0].value_counts().iloc[:40].index) 
plt.xticks (rotation=90)
plt.xlabel('Tokens')
plt.title('Freqeuncy vs Token  with stopwords for T1')
plt.savefig('freq_1.jpg')
plt.show()

#Forming wordcloud for the sherlock book
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,colormap='winter').generate(Sherlock_book) 
plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig('wordcloud_1.jpg')
plt.show() 


#Creating Stopwords
stop_words = set(stopwords.words('english'))
stop_words_2=set(STOPWORDS)
#Removing Stopwords from Sherlock tokens
S_tokens_sw = [i for i in S_tokenized if not i in stop_words and not i in stop_words_2]
S_text_sw = " "
S_text_sw = S_text_sw.join(S_tokens_sw) 


#Forming wordcloud for the sherlock book with removed stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='black', 
                min_font_size = 10,colormap='winter').generate(S_text_sw) 
plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig('wordcloud_2.jpg')
plt.show() 


#Finding frequency distrubution of tokens with removed stopwords 
S_freq=FreqDist(S_tokens_sw)
df_new=pd.DataFrame(S_tokens_sw)

#Plotting datframe of tokens with removed stopwords
sns.countplot(x=df_new[0], order=df_new[0].value_counts().iloc[:40].index) 
plt.xticks (rotation=90)
plt.xlabel('Tokens')
plt.title('Freqeuncy vs Token without stopwords for T1')
plt.savefig('tokenfreq_1.jpg')
plt.show()

#Finding wordlength 
word_len_S=[len(w) for w in S_tokenized]

#finding frequency of wordlength
freq_length=FreqDist(word_len_S)

#Converting word length list into dataframe 
df_len=pd.DataFrame(word_len_S)

#Plotting dataframe of length
sns.countplot(x=df_len[0]) 
plt.xlabel('Length of words')
plt.title('count vs word length for T1')
plt.savefig('len.jpg')
plt.show()

#Forming list of words For POS tagging
S_tag=S_text_sw.split(' ')
tagged=nltk.pos_tag(S_tag)

#Forming dictionary of POS-tagging
di={}
for i in tagged:
     di[i] = di.get(i, 0) + 1 

#Taking first 40 from dictionary of POS-tagging
d={}
d = (dict(list(di.items())[0:40]))
 
#Taking keys of the reduced dictionary
keys=list(d.keys())

#Taking tags from above keys
x=[k for i,k in keys]
#Taking values from reduced dictionary
values=d.values()

#Plotting tag vs frequency
plt.bar(x,values)
plt.title(' Frequency vs Tags for a few tags of T1')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.savefig('tagfreq.jpg')
plt.show()

#Forming dataframe of words and corresponding tags for whole dictionary 
df_tags = pd.DataFrame(di.keys()) 

#Taking first 40 entries of the dataframe and saving them as a csv file
df_reduced_tags = df_tags.head(41)
df_reduced_tags.to_csv('table_out.csv')
    

#############################################################################


#Reading Pride and Prejudice Book
Pride_book = open(r"Pride.txt",encoding='utf-8')
Pride_book=Pride_book.read()

#Cleaning The Pride and Prejudice book
Pride_book=re.sub(r"http\S+", "", Pride_book,flags=re.MULTILINE)#For http
Pride_book=re.sub(r' www\S+',' ', Pride_book) #For www.
Pride_book=re.sub(r'[^(a-zA-Z)\s]','',Pride_book) #For charcters other than alphabets
Pride_book=re.sub(r' (\n)',' ',Pride_book)#For removing sentnce gaps
Pride_book=re.sub(r' ','  ',Pride_book) #For removing extra space
Pride_book=re. sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b','',Pride_book)#For chapter names

#Tokenizing Pride and Prejudice Book
P_tokenized=word_tokenize(Pride_book)

#Finding frequency distrubution of token
P_freq=FreqDist(P_tokenized)


#Converting tokenized values into dataframe
df1=pd.DataFrame(P_tokenized)
#Plotting the tokens
sns.countplot(x=df1[0], order=df1[0].value_counts().iloc[:40].index) 
plt.xticks (rotation=90)
plt.xlabel('Tokens')
plt.title('Freqeuncy vs Token  with stopwords for T2')
plt.savefig('P_freq_1.jpg')
plt.show()

#Forming wordcloud for the Pride and Prejudice book
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,colormap='winter').generate(Pride_book) 
plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig('P_wordcloud_1.jpg')
plt.show() 


#Removing Stopwords from Pride and Prejudice tokens
P_tokens_sw = [i for i in P_tokenized if not i in stop_words and not i in stop_words_2]
P_text_sw = " "
P_text_sw = P_text_sw.join(P_tokens_sw) 


#Forming wordcloud for the Pride and Prejudice book with removed stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='black', 
                min_font_size = 10,colormap='winter').generate(P_text_sw) 
plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig('P_wordcloud_2.jpg')
plt.show() 


#Finding frequency distrubution of tokens with removed stopwords 
P_freq=FreqDist(P_tokens_sw)
df_new1=pd.DataFrame(P_tokens_sw)

#Plotting datframe of tokens with removed stopwords
sns.countplot(x=df_new1[0], order=df_new1[0].value_counts().iloc[:40].index) 
plt.xticks (rotation=90)
plt.xlabel('Tokens')
plt.title('Freqeuncy vs Token without stopwords for T2')
plt.savefig('P_tokenfreq_1.jpg')
plt.show()

#Finding wordlength 
word_len_P=[len(w) for w in P_tokenized]

#finding frequency of wordlength
freq_length1=FreqDist(word_len_P)

#Converting word length list into dataframe 
df_len1=pd.DataFrame(word_len_P)

#Plotting dataframe of length
sns.countplot(x=df_len1[0]) 
plt.xlabel('Length of words')
plt.title('count vs word length for T2')
plt.savefig('P_len.jpg')
plt.show()

#Forming list of words For POS tagging
P_tag=P_text_sw.split(' ')
tagged=nltk.pos_tag(P_tag)

#Forming dictionary of POS-tagging
di1={}
for i in tagged:
     di1[i] = di1.get(i, 0) + 1 

#Taking first 40 from dictionary of POS-tagging
d1={}
d1 = (dict(list(di1.items())[0:40]))
 
#Taking keys of the reduced dictionary
keys1=list(d1.keys())

#Taking tags from above keys
x1=[k for i,k in keys1]
#Taking values from reduced dictionary
values=d1.values()

#Plotting tag vs frequency
plt.bar(x1,values)
plt.title(' Frequency vs Tags for a few tags of T2')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.savefig('P_tagfreq.jpg')
plt.show()

#Forming dataframe of words and corresponding tags for whole dictionary 
df_tags1 = pd.DataFrame(di1.keys()) 

#Taking first 40 entries of the dataframe and saving them as a csv file
df_reduced_tags1 = df_tags1.head(41)
df_reduced_tags1.to_csv('P_table_out.csv')