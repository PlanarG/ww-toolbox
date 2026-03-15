import requests
import re
import json
import uuid
import os
base_url = "https://mc.appfeng.com"

html = requests.get(base_url + "/monster").text

content = re.findall(r'<div class="title">套装</div>(.+?)</div></div>', html)

linkAndNames = re.findall(r'<img src="(.*?)"> (.*?)<', content[0])

metadata = []

force = False

for (link, name) in linkAndNames:
    full_link = base_url + link
    print(full_link)
    img_name = os.path.basename(link)
    metadata.append({
        "name": name,
        "image": f"{img_name}"
    })
    if force == False and os.path.exists(img_name):
        continue
    img_data = requests.get(full_link).content
    with open(f"{img_name}", "wb") as img_file:
        img_file.write(img_data)

with open("metadata.json", "w", encoding="utf-8") as meta_file:
    json.dump(metadata, meta_file, ensure_ascii=False, indent=4)
    
print(linkAndNames)