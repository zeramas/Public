print("\nGender Analysis of Tweets")
print("Loading ML Libraries ... please wait")

import re
import pandas as pd
from nltk import trigrams, NaiveBayesClassifier, classify, word_tokenize, pos_tag, FreqDist
from sklearn.model_selection import train_test_split
from collections import Counter

MAX_TWEET_LEN = 240

def scrubTweet(twt):
    twt = re.sub("[^a-zA-Z]", ' ', twt.lower())
    return twt

def GenerateWordUsage(df):
    
    maleWords   = set()
    femaleWords = set()

    for row in df.itertuples():   
        
        gender = row.gender

        twt = scrubTweet(row.tweet)
        
        wordList = twt.split()
        for eachWord in wordList:
            if len(eachWord) >= 3 and len(eachWord) <= 12:
                if 'female' in gender:
                    femaleWords.add(eachWord)                
                elif 'male' in gender:
                    maleWords.add(eachWord)

    maleWords, femaleWords = maleWords-femaleWords, femaleWords-maleWords
    
    return maleWords, femaleWords    
    
def getFeatures(gender, twt, maleWords, femaleWords):
    
    ''' Model for feature extraction
        This is just and example not the real features
        that will be included.
    '''
    
    twtLength  = len(twt)
    wordList = twt.split()
    wordListLen = len(wordList)
    words = word_tokenize(twt)
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

    for word, tag in tags:
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
    
    features = {}
    features['tweetLength']       = twtLength
    features['wordListLen']        = wordListLen
    features['tweetRatio']         = twtLength / MAX_TWEET_LEN
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
      
    
    return features


def main():
    
    df = pd.read_csv("tweetsClean.csv")
    dfTrain, dfTest = train_test_split(df, test_size = 0.1, random_state=11)
    maleWords, femaleWords = GenerateWordUsage(dfTrain)    
    
    trainingFeatureSet = []
    testingFeatureSet  = []
    
    for row in dfTrain.itertuples():   
        gender = row.gender
        tweet  = scrubTweet(row.tweet)
        rowFeatures = getFeatures(gender, tweet, maleWords, femaleWords)
        trainingFeatureSet.append((rowFeatures, gender))
        
    for row in dfTest.itertuples():  
        gender = row.gender
        tweet  = scrubTweet(row.tweet)
        rowFeatures = getFeatures(gender, tweet, maleWords, femaleWords)
        testingFeatureSet.append((rowFeatures, gender))    
        
    # Create a NaiveBayes Gender Classifer from the Training Set
    genderClassifer = NaiveBayesClassifier.train(trainingFeatureSet)
    
    print('TrainSet Accuracy: ',classify.accuracy(genderClassifer, trainingFeatureSet)) 
    print('TestSet  Accuracy: ',classify.accuracy(genderClassifer, testingFeatureSet)) 
    
    genderClassifer.show_most_informative_features(20)
    
    
    
if __name__ == '__main__':
    main()
