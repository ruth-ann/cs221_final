import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfid
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys, json
from sklearn.metrics import silhouette_score
import numpy as np
import statistics 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import random
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize



def read_art_reviews():
    with open('artsReviewsTruncated.json') as f:
      artReviews = json.load(f)
    return artReviews

def read_science_reviews():
    with open('scienceReviewsTruncated.json') as f:
      scienceReviews = json.load(f)
    return scienceReviews

def read_music_reviews():
    with open('musicReviewsTruncated.json') as f:
      musicReviews = json.load(f)
    return musicReviews

def read_beauty_reviews():
    with open('beautyReviewsTruncated.json') as f:
      beautyReviews = json.load(f)
    return beautyReviews

def read_sports_reviews():
    with open('sportsReviewsTruncated.json') as f:
      sportsReviews = json.load(f)
    return sportsReviews

def read_all_reviews():
    with open('allReviewsTruncated.json') as f:
      allReviews = json.load(f)
    return allReviews

def read_art_dict():
    with open('artDict.json') as f:
      artDict = json.load(f)
    return artDict

def read_science_dict():
    with open('scienceDict.json') as f:
      scienceDict = json.load(f)
    return scienceDict

def read_music_dict():
    with open('musicDict.json') as f:
      musicDict = json.load(f)
    return musicDict

def read_beauty_dict():
    with open('beautyDict.json') as f:
      beautyDict = json.load(f)
    return beautyDict

def read_sports_dict():
    with open('sportsDict.json') as f:
      sportsDict = json.load(f)
    return sportsDict

artReviews = read_art_reviews()
scienceReviews = read_science_reviews()
musicReviews = read_music_reviews()
beautyReviews = read_beauty_reviews()
sportsReviews = read_sports_reviews()
allReviews = read_all_reviews()


art = ["machine", "sewing", "paper", "make", "set", "thread", "new", "old", "scissors", "book"]
music = ["sound", "guitar", "strings", "quality", "play", "nice", "better", "music", "playing", "sounds"]
beauty = ["hair", "skin", "use", "time", "using", "smell", "dry", "years", "scent", "long"]
science = ["used", "work", "gloves", "bit", "drill", "made", "need", "using", "tape", "tool"]
sports = ["bike", "back", "little", "light", "watch", "put", "fit", "still", "bag", "looking"]

def most_frequent_words(reviews):
    punctuation = '''!()-[]\{\};:'"\, <>./?@#$%^&*_~0123456789'''
    allWords = []
    # stopWords = stopwords.words() + ["great", "would", "buy", "good", "like", " n't", "n't", "really", " 'm", "'m", "well", "love", " 's", "'s", "get", "even", "much", "'ve", "bought", "price", "recommend", "could", "''", "product",  "...", "got", "easy", "``", "first"]    
    # stopWords = stopwords.words() + ["great", "would", "buy", "good", "use", "like", " n't", "n't", "use", "really", " 'm", "'m", "well", "love", " 's", "'s", "get", "used", "even", "much", "'ve", "new", "old", "bought", "price", "recommend", "could", "''", "product", "work", "time", "little", "...", "got", "easy", "``", "first", "--"]
    stopWords = stopwords.words()
    for review in reviews:
        words = word_tokenize(review.lower())

        wordsWithoutStopwords = [word for word in words if not word in stopWords and not word in punctuation]
        allWords+= wordsWithoutStopwords
    c = Counter(allWords)
    print(c.most_common(20))

def cluster(reviews, art, music, beauty, science, sports):
        assignments = []
        clusters = {0:[],1:[], 2:[], 3:[], 4:[]}
        for r in reviews:
            review = word_tokenize(r.lower())
            print(r)
            n = 5
            for word in review:
                if word in art:
                    # print(review)
                    # print(r)
                    clusters[0].append(r)
                    n = 0
                    break
            if n == 5:
                for word in review:
                    if word in music:
                        clusters[1].append(r)
                        n = 1
                        break
            if n == 5:   
                for word in review:
                    if word in sports:
                        clusters[2].append(r)
                        n = 2
                        break
            if n == 5:
                for word in review:
                    if word in beauty:
                        clusters[3].append(r)
                        n = 3
                        break
            if n == 5:
                for word in review:
                    if word in science:
                        clusters[4].append(r)
                        n = 4
                        break
    
            if n == 5:
                n = random.randint(0,4)
                clusters[n].append(r)
            assignments.append(n)
        return clusters, assignments

def construct_cluster_dictionary(reviews, labels, k):
    clusterDict = {}
    for i in range(k):
        clusterDict[i] = []
    for i in range(len(reviews)):
        clusterDict[labels[i]].append(reviews[i])
    return clusterDict

def analyze_clusters(reviews, labels, dictionary):
    categoryLabels = []
    labels = ['arts', 'sports', 'music', 'beauty', 'science']
    proportions = []
    for key in dictionary.keys():
        counts = dict.fromkeys(labels, 0)
        total = 0
        cluster = dictionary[key]
        for review in cluster:
            total += 1
            if review in artReviews:
                counts['arts'] += 1
                # categoryLabels.append('arts')
                categoryLabels.append(0.0)

            if review in sportsReviews:
                counts['sports'] += 1
                # categoryLabels.append('sports')
                categoryLabels.append(1.0)

            if review in musicReviews:
                counts['music'] += 1
                # categoryLabels.append('music')
                categoryLabels.append(2.0)

            if review in beautyReviews:
                counts['beauty'] += 1
                # categoryLabels.append('beauty')
                categoryLabels.append(3.0)

            if review in scienceReviews:
                counts['science'] += 1
                # categoryLabels.append('science')
                categoryLabels.append(4.0)
        topKey = max(counts, key=counts.get)
        proportions.append((key, topKey, counts[topKey]/total, total))
    return proportions, categoryLabels


def create_dataframe(reviews):
    vectorizer = tfid(
        min_df = 0.2,
        max_df = 0.95,
        max_features = 100000,
        stop_words = 'english'
    )
    vector = vectorizer.fit_transform(reviews)
    featureNames = vectorizer.get_feature_names()
    denselist = vector.todense().tolist()
    dataframe = pd.DataFrame(denselist, columns=featureNames) 

    return dataframe

def evaluate(reviews, assignments, dictionary):
    df = create_dataframe(allReviews)
    silhouetteScore = silhouette_score(df, assignments, metric='euclidean')
    print(silhouetteScore)
    return analyze_clusters(reviews, assignments, dictionary)


most_frequent_words(artReviews)
most_frequent_words(musicReviews)
most_frequent_words(sportsReviews)
most_frequent_words(beautyReviews)
most_frequent_words(scienceReviews)
dictionary, assignments = cluster(allReviews, art, music, beauty, science, sports)
print(assignments)
print(dictionary)
proportions, categoryLabels = evaluate(allReviews, assignments, dictionary)
print(proportions)

 
