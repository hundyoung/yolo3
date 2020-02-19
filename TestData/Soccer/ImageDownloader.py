import requests
import urllib
import os
import re
from bs4 import BeautifulSoup
import json


save_path = "./foul/"
url="https://images.api.press.net/api/v2/search/?category=A,S,E&ck=public&cond=not&crhPriority=1&fields_0=all&fields_1=all&imagesonly=1&limit=2000&orientation=both&page=1&q=football+foul&words_0=all&words_1=all"
response = requests.get(url)
print(response.text)
json = json.loads(response.text)
resultList = json["results"]
save_count =0
unsave_count =0
for result in resultList:
    # fileName=result["description_text"]
    renditions = result["renditions"]
    sampleSize = renditions["sample"]
    href = sampleSize["href"]
    fileName = str(href).split('/')[-1]
    if not os.path.exists(save_path+fileName):
        print(href)
        try:
            urllib.request.urlretrieve(href,save_path+fileName)
            save_count += 1
        except:
            print("Fail")
    else:
        unsave_count+=1

print("save",save_count,"images")
print("exists",unsave_count,"images")
# soup = BeautifulSoup(html.text, "html.parser")
# image_list = soup.find_all('img')

# for img in image_list:
#     print(img)
