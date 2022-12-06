'''
Identify images from a webpage using BeautifulSoup
Extract them using requests and display them using PIL
'''

# Python Standard Libaries
import requests                         # Python Standard Library for url requests
import os

# Python 3rd Party Libraries
from bs4 import BeautifulSoup           # 3rd Party BeautifulSoup Library - pip install Beautifulsoup4

url = 'urlhere'
base = 'urlhere'
IMG_SAVE = "./IMAGES/"  # Directory to store images

# Create the directory if necessary
if not os.path.exists(IMG_SAVE):
    os.makedirs(IMG_SAVE)

page = requests.get(url)       # retrieve a page from your favorite website
soup = BeautifulSoup(page.text, 'html.parser')  # convert the page into soup

print("Extracting Images from: ", url)
print("Please Wait")

images = soup.findAll('img')  # Find the image tags
for eachImage in images:      # Process and display each image
    try:
        imgURL = eachImage['src']
        print("Processing Image:", imgURL, end="")
        if imgURL[0:4] != 'http':       # If URL path is relative
            imgURL = base+imgURL         # try prepending the base url

        response = requests.get(imgURL)                 # Get the image from the URL
        imageName = os.path.basename(imgURL)
        
        imgOutputPath = IMG_SAVE+imageName
        
        with open(imgOutputPath, 'wb') as outFile:
            outFile.write(response.content)
            
        # Save the image
        print("  >> Saved Image:", imgOutputPath)
    except Exception as err:
        print(imgURL, err)
        continue    
print('\n\nScript Complete')
