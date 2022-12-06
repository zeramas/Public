139 lines (114 sloc)  3.75 KB
'''
Creates CSVs of Hue and Saturation data for a JPG
'''

#Imports
import os
from prettytable import PrettyTable
from PIL import Image
import pandas as pd
Image.MAX_IMAGE_PIXELS = 200000000  

#Global Variables
##Folder Names
REALIMGS = "REAL"
FAKEIMGS = "FAKE"

#Master DF
MASTERDF = pd.DataFrame(columns=['Image', 'Validity', 'HueChildDF', 'SatChildDF'])


#Helper Functions
def calcHueSat(pixel):
    ''' Calulate the
        Hue and Sat of a single pixel   
    '''
    r=pixel[0]
    g=pixel[1]
    b=pixel[2]      
    
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        hue = 0
    elif mx == r:
        hue = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        hue = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        hue = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        sat = 0
    else:
        sat = (df/mx)*100

    return hue, sat

def createIMGDF(imgFile):
    tblHue = PrettyTable(['0', '1', '2', '3', '4', '5', '6'])
    tblSat = PrettyTable(['0', '1', '2', '3', '4', '5', '6'])

    
    # df.rename(columns={x:y for x,y in zip(df.columns,range(0,len(df.columns)))})
    
    print("Image Feature Extraction - Basics")
    # imgFile = "REAL/barn.jpg"
    print(f"Opening {imgFile}")
    img = Image.open(imgFile)  
    maxRows = img.width-1
    maxCols = img.height-1
    # print(range(maxCols))
    HueDF = pd.DataFrame(columns=range(maxCols))
    SatDF = pd.DataFrame(columns=range(maxCols))
    
    pix = img.load()    
    
    for row in range(0,maxRows):
        rowHue = []
        rowSat = []        
        for col in range(0,maxCols):
            testPixel = pix[row, col]
            # print(f"{row},{col}")
            hue, sat = calcHueSat(testPixel)
            rowHue.append(round(hue, 2))
            rowSat.append(round(sat, 2))

        HueDF.loc[len(HueDF)] = rowHue
        SatDF.loc[len(SatDF)] = rowSat
    

    return HueDF, SatDF

#Main
if __name__ == "__main__":

    for file in os.listdir(REALIMGS):
        filename = os.fsdecode(file)
        # print(os.path.join(str(REALIMGS), filename))
        imgFile = os.path.join(str(REALIMGS), filename)
        picname, filext = filename.split(".")
        HueDF, SatDF = createIMGDF(imgFile)
        newfp, trash = imgFile.split(".")
        HueCSVName = newfp+"Hue"+".csv"
        SatCSVName = newfp+"Sat"+".csv"
        print(f"Saving {HueCSVName}...")
        HueDF.to_csv(HueCSVName)
        print(f"Saving {SatCSVName}...")
        SatDF.to_csv(SatCSVName) 
        #Appends info to MASTERDF
        print("Appending item to MASTERDF...")
        MASTERDF.loc[len(MASTERDF)] = picname, 'Real', HueCSVName, SatCSVName   
        
    
    for file in os.listdir(FAKEIMGS):
        filename = os.fsdecode(file)
        # print(os.path.join(str(FAKEIMGS), filename))
        imgFile = os.path.join(str(FAKEIMGS), filename)
        picname, filext = filename.split(".")
        HueDF, SatDF = createIMGDF(imgFile)
        newfp, trash = imgFile.split(".")
        HueCSVName = newfp+"Hue"+".csv"
        SatCSVName = newfp+"Sat"+".csv"
        print(f"Saving {HueCSVName}...")
        HueDF.to_csv(HueCSVName)
        print(f"Saving {SatCSVName}...")
        SatDF.to_csv(SatCSVName) 
        #Appends info to MASTERDF
        print("Appending item to MASTERDF...")
        MASTERDF.loc[len(MASTERDF)] = picname, 'FAKE', HueCSVName, SatCSVName


print(MASTERDF)
MASTERDF.to_csv('MASTERDF.csv')

### Can be used to combine all data into ONE DF
MASTERDF = pd.read_csv("MASTERDF.csv")

for row, col in MASTERDF.iterrows():
    # print("row:",row)
    print("col:",col['Image'])
    hueDF = pd.read_csv(col['HueChildDF'])
    SatDF = pd.read_csv(col['SatChildDF'])
    MASTERDF = pd.concat([MASTERDF,hueDF,SatDF])

#print(MASTERDF)
