import nltk
from nltk.tokenize import sent_tokenize

text = "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization"


# --------- sentence tokenization --------- 

tokenized_text = sent_tokenize(text)
print("💖 Tokenized text: \n ", tokenized_text)


# ---------  word tokenization --------- 

from nltk.tokenize import word_tokenize 

tokenized_word = word_tokenize(text)
print("💖 Tokenized word: \n", tokenized_word)


# ---------  print stop words of English --------- 

from nltk import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
print("💖 Stop words are: \n", stop_words) 

text = "How to remove stop words with NLTK library in Python?" 
text = re.sub('[^a-zA-Z]', ' ',text) 
tokens = word_tokenize(text.lower())
 
filtered_text = [] 

for w in tokens: 
    if w not in stop_words: 
        filtered_text.append(w) 
        
print("💖 Tokenized Sentence: \n",tokens) 
print("💖 Filterd Sentence: \n",filtered_text) 


# ---------  Perform Stemming --------- 

from nltk.stem import PorterStemmer
 
e_words = ["wait", "waiting", "waited", "waits"] 
ps = PorterStemmer() 

for w in e_words: 
    rootWord = ps.stem(w) 
    
print("💖 Rootword: \n", rootWord) 


# --------- Perform Lemmatization --------- 

from nltk.stem import WordNetLemmatizer 

wordnet_lemmatizer = WordNetLemmatizer() 
text = "studies studying cries cry"

tokenization = nltk.word_tokenize(text)
 
for w in tokenization: 
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w))) 


# --------- Apply POS Tagging to text  --------- 

data = "The pink sweater fit her perfectly" 
words = word_tokenize(data) 
for word in words: 
    print("💖", nltk.pos_tag([word])) 