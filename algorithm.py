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


def most_frequent_words(reviews):
    punctuation = '''!()-[]\{\};:'"\, <>./?@#$%^&*_~0123456789'''
    allWords = []
    stopWords = stopwords.words() + ["great", "would", "buy", "good", "use", "like", " n't", "n't", "use", "really", " 'm", "'m", "well", "love", " 's", "'s", "get", "used", "even", "much", "'ve", "new", "old", "bought", "price", "recommend", "could", "''", "product", "work", "time", "little", "...", "got", "easy", "``", "first"]
    for review in reviews:
        words = word_tokenize(review.lower())

        wordsWithoutStopwords = [word for word in words if not word in stopWords and not word in punctuation]
        allWords+= wordsWithoutStopwords
    c = Counter(allWords)
    print(c.most_common(20))



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
                categoryLabels.append(0.0)

            if review in sportsReviews:
                counts['sports'] += 1
                categoryLabels.append(1.0)

            if review in musicReviews:
                counts['music'] += 1
                categoryLabels.append(2.0)

            if review in beautyReviews:
                counts['beauty'] += 1
                categoryLabels.append(3.0)

            if review in scienceReviews:
                counts['science'] += 1
                categoryLabels.append(4.0)


            
            
        topKey = max(counts, key=counts.get)
        proportions.append((key, topKey, counts[topKey]/total, total))
    return proportions, categoryLabels
    


def best_params(reviews, k):
    score = []
    maxScore = 0
    run = []
    preds = []
    percentageList = [[] for i in range(5)]
    for i in range(1,10):
        df, data2D = create_dataframe(reviews, i)
        silhouetteScore, currPred = run_kmeans(df, k)
        dictionary = construct_cluster_dictionary(reviews, currPred, k)
        proportions, categoryLabels = analyze_clusters(reviews, currPred, dictionary)
        percentages = [elem[2] for elem in proportions]
        if(silhouetteScore > maxScore):
            maxScore = silhouetteScore
            preds = currPred
        run.append(i)
        score.append(silhouetteScore)
        
        print("\nMinDf = ",i)
        print(categoryLabels)
        print(proportions)

        for i in range(5):
            percentageList[i].append(percentages[i])
    plt.plot(run, score)
    for i in range(5):
        n = i%4
        plt.scatter(run, percentageList[i], alpha=0.7, marker = "{}".format(n+1))
    sys.stdout.close()

    plt.xlabel("min df")
    plt.ylabel("score")
    plt.title("Plot of min df against Scores")
    plt.savefig('mindfwhole.png')
    

    for i in range(1,5):
        print(i)
        df, data2D = create_dataframe(reviews, i  * 0.05)
        silhouetteScore, currPred = run_kmeans(df, k)
        dictionary = construct_cluster_dictionary(reviews, currPred, k)
        proportions, categoryLabels = analyze_clusters(reviews, currPred, dictionary)
        percentages = [elem[2] for elem in proportions]
        if(silhouetteScore > maxScore):
            maxScore = silhouetteScore
            preds = currPred
        run.append(i * 0.05)
        score.append(silhouetteScore)
        
        print("\nMinDf = ",i * 0.05)
        print(categoryLabels)
        print(proportions)
        for i in range(5):
            percentageList[i].append(percentages[i])
    plt.plot(run, score)
    for i in range(5):
        n = i%4
        plt.scatter(run, percentageList[i], alpha=0.7, marker = "{}".format(n+1))

    plt.xlabel("min df")
    plt.ylabel("score")
    plt.title("Plot of min df against Scores")
    plt.savefig('mindffrac.png')

    sys.stdout = open("kmeandiffk.txt", "a")


    for i in range(5,20):
        df, data2D = create_dataframe(reviews, 5)
        silhouetteScore, currPred = run_kmeans(df, i)
        dictionary = construct_cluster_dictionary(reviews, currPred, i)
        proportions, categoryLabels = analyze_clusters(reviews, currPred, dictionary)
        percentages = [elem[2] for elem in proportions]
        if(silhouetteScore > maxScore):
            maxScore = silhouetteScore
            preds = currPred
        run.append(i)
        score.append(silhouetteScore)
        
        print("\nK Value = ",i)
        print(categoryLabels)
        print(proportions)
        for i in range(5):
            percentageList[i].append(percentages[i])
    plt.plot(run, score)
    for i in range(5):
        n = i%4
        plt.scatter(run, percentageList[i], alpha=0.7, marker = "{}".format(n+1))
    sys.stdout.close()

    plt.xlabel("min df")
    plt.ylabel("score")
    plt.title("Plot of min df against Scores")
    plt.savefig('diffk.png')   

def kmeans_pipeline(reviews, k):
    df, data2D = create_dataframe(reviews, 5)
    labels = run_kmeans(df, k)
    silhouetteScore, labels = run_kmeans(df, k)
    dictionary = construct_cluster_dictionary(reviews, labels, k)
    proportions, categoryLabels = analyze_clusters(reviews, labels, dictionary)
    plt.figure(figsize=(8, 6))
    plt.scatter(data2D[:,0], data2D[:,1], c=labels.astype(float))
    plt.savefig('results.png')
    plt.figure(figsize=(8, 6))
    plt.scatter(data2D[:,0], data2D[:,1], c=categoryLabels)
    plt.savefig('true.png')
    return dictionary, labels, proportions

def construct_cluster_dictionary(reviews, labels, k):
    clusterDict = {}
    for i in range(k):
        clusterDict[i] = []
    for i in range(len(reviews)):
        clusterDict[labels[i]].append(reviews[i])
    return clusterDict

def run_kmeans(dataframe, k):
    run = []
    score = []
    maxScore = 0
    preds = []
    currPred = KMeans(n_clusters=k, init = 'k-means++', algorithm = 'elkan').fit_predict(dataframe)
    silhouetteScore = silhouette_score(dataframe, currPred, metric='euclidean')
    return silhouetteScore, currPred

def create_dataframe(reviews, minDf):
    vectorizer = tfid(
        min_df = 5,
        max_df = 0.95,
        max_features = 100000,
        stop_words = 'english'
    )
    vector = vectorizer.fit_transform(reviews)
    featureNames = vectorizer.get_feature_names()
    denselist = vector.todense().tolist()
    dataframe = pd.DataFrame(denselist, columns=featureNames) 
    pca = PCA(n_components=2).fit(vector.todense())
    data2D = pca.transform(vector.todense())

    return dataframe, data2D

def generate_clusters():
    print("Outputting clusters:")
    output_clusters(allReviews, 5)


def output_clusters(reviews, k):
    clusters, preds, proportions = kmeans_pipeline(reviews, k)
    sys.stdout = open("clusters{0}means.txt".format(k), "a")
    for k in clusters.keys():
        print("\n__________________________________")
        print("\nCluster ", k)
        print("\n__________________________________")
        elements = clusters[k]
        for i in range(len(elements)):
            print("\n")
            print(elements[i])
        print("\n")
    print(preds)
    print(proportions)



best_params(allReviews, 5)
generate_clusters()