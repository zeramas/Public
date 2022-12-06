# imports
from collections import Counter
import re
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords
import pandas as pd

#Library Setups
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 2000)
stops = set(stopwords.words('english'))

#Global Variables
DEBATE_FILE = "speechByCandidate.csv"    
CANDIDATES = ['joe biden', 'bernie sanders', 'tulsi gabbard', 'elizabeth warren', 'michael bloomberg', 'amy klobuchar',
              'pete buttigieg', 'tom steyer', 'michael bennet', 'andrew yang', 'john delaney', 'cory booker',
              'marianne williamson', 'kamala harris', 'steve bullock', 'tim ryan',
              'bill de blasio', 'kirsten gillibrand', 'jay inslee', 'eric swalwell']

topWords = {}


# take second element for sort
def CountElement(elem):
    return elem[1]

def topwords():
    with open(DEBATE_FILE) as debate:
        for eachLine in debate:
            
            eachLine = eachLine.lower()
            lineList = eachLine.split(',')
            wordList = lineList[1].split()
            
            for eachWord in wordList:
                
                if eachWord not in stops and len(eachWord) >= 5:
                    try:
                        cnt = topWords[eachWord]
                        cnt += 1
                        topWords[eachWord] = cnt
                    except:
                        topWords[eachWord] = 1
                        
        topWordList = list(topWords.items())
        topWordList.sort(key=CountElement, reverse=True)
        
        print(topWordList[0:50])
        twl = topWordList[0:50]
        newdf = pd.DataFrame(data=topWords.items())
        newdf = newdf.rename(columns={0:'Word',1:'Overall Occurance'})
        newdf = newdf.sort_values(by=['Overall Occurance'],ascending=False)
        #newdf = newdf.set_index('Word').transpose()#.reset_index()
        #newdf = newdf.rename(columns={'index':'Word'})
        #newdf = newdf.rename_axis('index')
        topfifty = newdf.head(50)
        return topfifty, twl



def candidateWords(candidate):

    candidateText = '' 
    
    with open(DEBATE_FILE) as debate:
        for eachLine in debate:
            eachLine = eachLine.lower()

            nameEnd = eachLine.find(',')
            name = eachLine[0:nameEnd]
            scrubbedName = re.sub("[^a-zA-z ]", ' ', name)

            if candidate in eachLine:
                candidateText = candidateText + eachLine[nameEnd+1:] 

    candidateWords = word_tokenize(candidateText)
      

    

    return candidateWords


topfifty, twl = topwords()
print('\nStep one:\n')
print(f'\nHere are the top fifty words used across all candidates:\n\n{topfifty}\n\n')
print('\nStep Two:\n')


candwordsDF = pd.DataFrame(columns=['Candidate', 'Candidate Words'])
candidatewordslist = []

def main():
    runner = []
    for candidate in CANDIDATES:
        runnerdict = {}
        runnerdict['candidate'] = candidate
        
        words = candidateWords(candidate)
        for t in twl:
            t = t[0]
            runnerdict[t] = 0
            for word in words:
                if t == word:
                    cnt = runnerdict[t]
                    cnt += 1
                    runnerdict[t] = cnt
                    
                else:
                    continue

        runner.append(runnerdict)
    
    
    return topfifty, runner                
                

topfiftyy, runner = main()

runnerdf = pd.DataFrame.from_records(runner)
print(runnerdf)
#save to csv
