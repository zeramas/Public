
#Imports
from prettytable import PrettyTable
import pandas as pd
import re


'''
Class wordcounter can;
-take an input text filepath
-normalize speech
-remove stopwords
-extract occurances of each word
-display an output dataframe ['word','word_occur','word_freq'] #df.head(25)#by_asc
--optionally save df as csv
'''


# Prepare a STOP_WORD List
with open("STOP_WORDS.txt", 'r') as stops:
    stopWords = stops.read()

#Global Var
STOP_WORDS = stopWords.split()


class wordcounter():

    def __init__(self,textfile):
        self.textfile = textfile #Create text var for class instance
        self.bigdf = pd.DataFrame(columns=['Word','Word Occurance','Word Frequency'])#Create blank df for class instance

    def parsefile(self):
        print(f"Reading text file {self.textfile}...")
        with open(self.textfile, 'r') as book:
            text = book.read()
        
        print(f'Formating text file...')
        text = text.lower()
        
        content = re.sub("[^a-z]", ' ', text)

        wordList = content.split()
        print(f'Word list created!')
 
        wordCnt = 0

        wordDict = {}
        print(f'Creating table and sorting by Word Occurance!')
        for eachWord in wordList:
            # ignore articles and STOP_WORDS
            if eachWord in STOP_WORDS or len(eachWord) <= 3:
                continue
            try:
                wordCnt += 1
                cnt = wordDict[eachWord]
                cnt += 1
                wordDict[eachWord] = cnt
            except:
                wordDict[eachWord] = 1

        for word, occurrence, in wordDict.items():
            freq = round(((occurrence/wordCnt) * 100.0), 2)
            self.bigdf.loc[len(self.bigdf.index)] = [word, occurrence, freq]
                  
        self.bigdf = self.bigdf.sort_values(by='Word Occurance', ascending=False)
        
        return self.bigdf


 
#List of all speeches to process
speechlist = ["SPEECH1.txt","SPEECH2.txt","SPEECH3.txt","SPEECH4.txt","SPEECH5.txt"]

#Iterate through all speeches, saves a csv output to local directory
for speech in speechlist:
    print(f'Processing {speech}...')
    sp = wordcounter(speech)
    sp = sp.parsefile()
    print(sp.head(25))
    sp.to_csv(f'{speech}.csv')
