import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Define two documents
documentA = 'Jupiter is the largest Planet'
documentB = 'Mars is the fourth planet from the Sun'

# Tokenization: Split each document into individual words
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

# Create a set of unique words by combining the bags of words from both documents
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

# Initialize dictionaries to store word frequencies in each document
numOfWordsA = dict.fromkeys(uniqueWords, 0)
numOfWordsB = dict.fromkeys(uniqueWords, 0)

# Count word frequencies in documentA
for word in bagOfWordsA:
    numOfWordsA[word] += 1

# Count word frequencies in documentB
for word in bagOfWordsB:
    numOfWordsB[word] += 1

# Compute the term frequency (TF) for each word in the bag of words
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

# Compute TF for both documents
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)

# Compute the inverse document frequency (IDF) for each word
def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

# Compute IDF for both documents
idfs = computeIDF([numOfWordsA, numOfWordsB])

print("💖 idfs: m idfs \n", idfs)

# Compute the term TF-IDF for all words
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

# Compute TF-IDF for both documents
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

# Display TF-IDF scores as a DataFrame
df = pd.DataFrame([tfidfA, tfidfB])
print("💖: \n", df)
