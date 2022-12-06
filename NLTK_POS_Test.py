# imports
from collections import Counter
import re
from nltk import word_tokenize, pos_tag, FreqDist
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# Global Variables
DEBATE_FILE = "debateRaw.txt"
#CANDIDATES = ['joe biden', 'tom steyer', 'tim ryan', 'jay inslee']
CANDIDATES = ['joe biden', 'bernie sanders', 'tulsi gabbard', 'elizabeth warren', 'michael bloomberg', 'amy klobuchar',
              'pete buttigieg', 'tom steyer', 'michael bennet', 'andrew yang', 'john delaney', 'cory booker',
              'marianne williamson', 'kamala harris', 'steve bullock', 'tim ryan',
              'bill de blasio', 'kirsten gillibrand', 'jay inslee', 'eric swalwell']
MODERATORS = ['dana bash', 'don lemon', 'jake tapper']
'''
Step One: Pre-process the text file to create a CSV file that contains two columns - Candidate, and Candidate Response
--Will need to research all candidates, and moderators. Response row == all responses concat'd into one string
Step Two: Using NLTK and PD extract the language elements used by each candidate using the 'part of speech' method.
Each row of the df will include the following columns: Candidate, (columns for each language element)
--Only one row per candidate, values for each field will be number of times language element was used
'''

def main(candidate):

    candidateText = '' 
    print(f"Processing {candidate} Results...")
    with open(DEBATE_FILE) as debate:
        for eachLine in debate:
            eachLine = eachLine.lower()

            nameEnd = eachLine.find(',')
            name = eachLine[0:nameEnd]
            scrubbedName = re.sub("[^a-zA-z ]", ' ', name)

            if candidate in eachLine:
                candidateText = candidateText + eachLine[nameEnd+1:] 

    candidateWords = word_tokenize(candidateText)

    print(f"Returning Tokens for {scrubbedName}.")

    return candidateWords

resultsDF = pd.DataFrame(columns=['Candidate', 'Candidate Response'])

def bigfunc():
    runner = []
    for candidate in CANDIDATES:
        resultsDF.loc[len(resultsDF.index)] = candidate, main(candidate)
        tags = pos_tag(main(candidate))
        counts = Counter(tag for word,  tag in tags)
        fd = FreqDist(counts)
        newdf = pd.DataFrame(fd.items(), columns=['Language element', 'Frequency'])
        newdf = newdf.set_index('Language element')
        newdf = newdf.transpose()
        newdf['Candidate'] = candidate
        runner.append(newdf)
        
    return runner,resultsDF


runner,resultsDF = bigfunc()
print('Printing resultsDF to CSV.')
resultsCSV = resultsDF.to_csv('resultsCSV.csv')
df = pd.concat(runner)
print(df)
dfcsv = df.to_csv('dfcsv.csv')
