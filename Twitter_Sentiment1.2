
import re
from collections import Counter

print("Twitter Sentiment")
print("Loading ML Libraries ..... Please Wait ...")
import pandas as pd
from nltk import word_tokenize, pos_tag, FreqDist, trigrams

# Setup Prettytable for results
from prettytable import PrettyTable

# Machine Learning Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Psuedo Constants
DEBUG = False   # set DEBUG = True, for debug messages

#Psuedo Lookup for positive and negative sentiments
SENTIMENT = {'Yes':1, 'No':0}

# Set Panda Options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)   
pd.set_option('display.width', 2000)   

# Create simplified dataframe for globalWarming.csv
# Just two columns, existence, tweet
print("\nCreating dataframe from globalWarming.csv ...")
df = pd.read_csv('globalWarming.csv', encoding = 'ISO-8859-1')
df = df.rename(columns={'existence': 'Sentiment', 'tweet': 'Tweet'})
df = df[['Sentiment','Tweet']]

def scrubTweet(twt):
    
    twt = re.sub("[^a-zA-Z.?!,]", ' ', twt)
    return twt

def genPosNegTriGrams(dfCheck):
    '''
    Process each row in the training dataframe
    '''
    posTrigrams = set()
    negTrigrams = set()
    
    for row in dfCheck.itertuples():
        if row.Sentiment != 'Yes' and row.Sentiment != 'No':
            continue        
        twt = twt = scrubTweet(row.Tweet)
        
        triGrams = list(trigrams(twt.split()))
        
        for tri in triGrams:
            if row.Sentiment == 'Yes':
                posTrigrams.add(tri)
            else:
                negTrigrams.add(tri)
    
    posTrigrams -= negTrigrams
    negTrigrams -= posTrigrams
    
    return posTrigrams, negTrigrams

def getFeatures(sentiment, twt, posG, negG):
    ''' Model for feature extraction
        This is just and example not the real features
        that will be included.
    '''
    twt = scrubTweet(twt)
    
    posTriCount = 0
    negTriCount = 0
    twtGrams = list(trigrams(twt.split()))
    trigramCnt = len(twtGrams)
    
    for eachGram in twtGrams:
        if eachGram in posG:
            posTriCount += 1
        if eachGram in negG:
            negTriCount += 1
    try:
        posTrigramPercentage = round((posTriCount/trigramCnt),2)
    except:
        posTrigramPercentage = 0.0
    
    try:
        negTrigramPercentage = round((negTriCount/trigramCnt),2)
    except:
        negTrigramPercentage = 0.0
    
    tokenizedWords = word_tokenize(twt)
    tokenWords     = [w for w in tokenizedWords]
    wordCnt        = len(tokenWords)
    
    pos_tagged = pos_tag(tokenWords)
    counts = Counter(tag for word,tag in pos_tagged)
    
    adjFreq = round(counts['JJ']/wordCnt,  4)    # Adjectives
    cmpFreq = round(counts['JJR']/wordCnt, 4)   # Adjectives Comparative
    supFreq = round(counts['JJS']/wordCnt, 4)   # Adjectives Superlative    

    return [SENTIMENT[sentiment], wordCnt, adjFreq, cmpFreq, supFreq, posTrigramPercentage, negTrigramPercentage]

def main():
    
    print("\nCreating dataframe from globalWarming.csv ...")
    df = pd.read_csv('globalWarming.csv', encoding = 'ISO-8859-1')
    df = df.rename(columns={'existence': 'Sentiment', 'tweet': 'Tweet', 'existence.confidence':'confidence'})
    df.dropna(inplace= True)
    df = df[df['confidence'] == 1] #Drop unwanted rows  
    df = df[['Sentiment','Tweet']] 
    
    featureList   = []  # List of features for each sample
    sentimentList = []  # Corresponding sentiment
        
    dfTrain, dfTest = train_test_split(df, test_size = 0.2, random_state=42)
        
    '''
    Get pos and neg trigrams sets from training dataset
    '''
    posTrigrams, negTrigrams = genPosNegTriGrams(dfTrain)
        
    print("Processing Training Dataframe ...")
    for row in df.itertuples():
        # only process rows that are either Yes or No Sentiment Values
        if row.Sentiment != 'Yes' and row.Sentiment != 'No':
            continue
        
        sentimentList.append(row.Sentiment)  # update the sentiment list
        features = getFeatures(row.Sentiment,row.Tweet, posTrigrams, negTrigrams)
        featureList.append(features)  # Update the corresponding feature list    

    dfModel = pd.DataFrame(featureList, columns=['SENTIMENT', 'WORDCNT', 'ADJ%','COMPARATIVE%','SUPERLATIVE%', 'POS-TRIGRAMS%', 'NEG-TRIGRAMS%'])
    dfModel.to_csv("GlobalWarmingFreq.csv")

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    plotVars = ['POS-TRIGRAMS%', 'NEG-TRIGRAMS%','ADJ%', 'COMPARATIVE%']
    pp = sns.pairplot(dfModel, vars=plotVars, height=3.5, palette="rainbow")
    plt.title("POS v NEG Trigram Pairplot")
    plt.show()  
    
    # Create a K Nearest Neighbor Classifier
    
    print("\n\nCreating Nearest Neighbor Classifer Model ...")
    scaler = StandardScaler()
    scaler.fit(dfModel.drop('SENTIMENT', axis=1))
    scaled_features = scaler.transform(dfModel.drop('SENTIMENT', axis=1))
    scaled_data = pd.DataFrame(scaled_features, columns = dfModel.drop('SENTIMENT', axis=1).columns)
    
    x = scaled_data
    y = dfModel['SENTIMENT']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(x_train, y_train)    
    
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))
       

if __name__ == '__main__':
    main()
    print("\nScript End")
