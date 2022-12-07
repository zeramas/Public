#Imports
import re
from collections import Counter
import pandas as pd
from nltk import word_tokenize, pos_tag, FreqDist, NaiveBayesClassifier, classify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set Panda Options
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 1000)   
# pd.set_option('display.width', 2000)

print("\nCreating dataframe from COMBINED_NEWS.csv ...")
df = pd.read_csv('COMBINED_NEWS.csv', encoding = 'ISO-8859-1')
df = df[['news_article', 'factuality']]

#Functions

def createCombinedNews():
    print("Reading FAKENEWS.csv...")
    fakenewsdf = pd.read_csv('FAKENEWS.csv', names=['news_article'])
    fakenewsdf = fakenewsdf.fillna("")
    fakenewsdf.dropna(inplace=True)
    fakenewsdf = fakenewsdf['news_article']
    fakenewsdf = fakenewsdf.to_frame()
    fakenewsdf['factuality'] = 'FAKE'
    print(fakenewsdf.head())

    print("Reading REALNEWS.csv...")
    realnewsdf = pd.read_csv('REALNEWS.csv', names=['news_article'])
    realnewsdf = realnewsdf.fillna("")
    realnewsdf.dropna(inplace=True)
    realnewsdf = realnewsdf['news_article']
    realnewsdf = realnewsdf.to_frame()
    realnewsdf['factuality'] = 'TRUE'
    print(realnewsdf.head())

    print("Creating COMBINED_NEWS.csv...")
    combinednewsdf = pd.concat([fakenewsdf,realnewsdf])
    ####
    ##Note:
    #I had to reduce the size to 10000 rows because my computer couldn't finish running the script
    combinednewsdfhead = combinednewsdf.head(5000)
    combinednewsdftail = combinednewsdf.tail(5000)
    combinednewsdf = pd.concat([combinednewsdfhead,combinednewsdftail])
    ####
    combinednewsdf.to_csv("COMBINED_NEWS.csv")
    print(combinednewsdf.head())
    print(combinednewsdf.tail())

    return combinednewsdf

def scrubNews(news):
    news = re.sub("[^a-zA-Z]", ' ', str(news))
    return news

def GenerateWordUsage(df):
     
    trueWords   = set()
    fakeWords = set()

    for row in df.itertuples():   
        
        factuality = row.factuality

        news = scrubNews(row.news_article)
        
        wordList = news.split()
        for eachWord in wordList:
            if len(eachWord) >= 3 and len(eachWord) <= 12:
                if 'FAKE' in factuality:
                    fakeWords.add(eachWord)                
                elif 'TRUE' in factuality:
                    trueWords.add(eachWord)
    wordIntersect = trueWords.intersection(fakeWords)
    trueWords, fakeWords = trueWords-wordIntersect, fakeWords-wordIntersect

    return trueWords, fakeWords  


def getFeatures(factuality, news, trueWords, fakeWords):
    

    newsLength  = len(news)
    wordList = news.split()
    wordListLen = len(wordList)
    words = word_tokenize(news)
    tags = pos_tag(words)
    counts = Counter(tag for word,  tag in tags)
    fd = FreqDist(counts)
    NNPcnt = 0
    CCcnt = 0
    INcnt = 0 #preposition####
    RBcnt = 0 # 		| Adverb |####
    JJRcnt = 0 # 		| Adjective, comparative |
    VBNcnt = 0 # 	| Verb, past participle |
    VBDcnt = 0 # 		| Verb, past tense |##
    PRPcnt = 0 # 		| Personal pronoun |###
    UHcnt = 0 # 		| Interjection |####
    VBGcnt = 0 # 		| Verb, gerund or present participle |
    CDcnt = 0 #Cardinal Number
    WPcnt = 0 # 		| Wh-pronoun |
    trueWordCnt   = 0
    fakeWordCnt = 0
    usedWordCnt = 0

    for word, tag in tags:
        usedWordCnt += 1
        if tag == 'NNP':
            NNPcnt += 1
        elif tag == 'CC':
            CCcnt += 1
        elif tag == 'IN':
            INcnt += 1
        elif tag == 'RB':
            RBcnt += 1
        elif tag == 'JJR':
            JJRcnt += 1
        elif tag == 'VBN':
            VBNcnt += 1
        elif tag == 'VBD':
            VBDcnt += 1
        elif tag == 'PRP':
            PRPcnt += 1
        elif tag == 'UH':
            UHcnt += 1
        elif tag =='VBG':
            VBGcnt += 1
        elif tag == 'CD':
            CDcnt += 1
        elif tag == 'WP':
            WPcnt += 1
        if word in trueWords:
            trueWordCnt += 1
        elif word in fakeWords:
            fakeWordCnt += 1
    
    features = {}
    features['newsLength']       = newsLength
    features['wordListLen']        = wordListLen
    features['nounRatio']          = NNPcnt / wordListLen
    features['conjunctionRatio']   = CCcnt / wordListLen
    features['prepositionRatio']   = INcnt / wordListLen
    features['adverbRatio']        = RBcnt / wordListLen
    features['adjCompRatio']       = JJRcnt / wordListLen
    features['verbPastRatio']      = VBNcnt / wordListLen
    features['verbPastTRatio']     = VBDcnt / wordListLen
    features['personalPronounRatio'] = PRPcnt / wordListLen
    features['interjRatio']        = UHcnt / wordListLen
    features['verbRatio']          = VBGcnt / wordListLen
    features['cardNumRatio']       = CDcnt / wordListLen
    features['wpronounRatio']      = WPcnt / wordListLen
    features['trueWordRatio']   = round(((trueWordCnt/usedWordCnt)*100.0), 4)
    features['fakeWordRatio'] = round(((fakeWordCnt/usedWordCnt)*100.0), 4)
    features['usedWords']          = usedWordCnt
      
    
    return features

#Main

def main():

    #Generate df from news
    df = createCombinedNews()

    #Seperate test and training data from main df
    print("Splitting CSV data into training and testing...")
    dfTrain, dfTest = train_test_split(df, test_size = 0.3, random_state=11)

    #Characterize word usage within the news as fake or true
    print("Characterizing word usage as fake or true...")
    trueWords, fakeWords = GenerateWordUsage(df)   
    
    #Generate feature lists
    trainingFeatureList = []
    testingFeatureList = []

    #Generate training features list
    print("Creating Training Features List...")
    for row in dfTrain.itertuples():   
        factuality = row.factuality
        news_article  = scrubNews(row.news_article)
        # print(f"Getting features for {row}...")
        rowFeatures = getFeatures(factuality, news_article, trueWords, fakeWords)
        # print(f"Appending features for {row}...")
        trainingFeatureList.append((rowFeatures, factuality))
        

    #Generate testing features list 
    print("Creating Testing Features List...") 
    for row in dfTest.itertuples():  
        factuality = row.factuality
        news_article  = scrubNews(row.news_article)
        rowFeatures = getFeatures(factuality, news_article, trueWords, fakeWords)
        testingFeatureList.append((rowFeatures, factuality))
           

    trainingFeatureListLen = len(trainingFeatureList)
    print(f"Training set size: {trainingFeatureListLen}")
    testingFeatureListLen = len(testingFeatureList)
    print(f"Testing set size: {testingFeatureListLen}") 

    # Create a NaiveBayes News Classifer from the Training Set
    print("Training Naive Bayes Classifier!")
    newsClassifer = NaiveBayesClassifier.train(trainingFeatureList)
    
    print('TrainSet Accuracy: ',classify.accuracy(newsClassifer, trainingFeatureList)) 
    print('TestSet  Accuracy: ',classify.accuracy(newsClassifer, testingFeatureList)) 
   
    newsClassifer.show_most_informative_features(20)

if __name__ == '__main__':
    main()
