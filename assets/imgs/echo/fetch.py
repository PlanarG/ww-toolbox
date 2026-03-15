import requests
import re
import json
import uuid
import os
base_url = "https://mc.appfeng.com"

html = requests.get(base_url + "/monster").text

names = re.findall(r'<div class="name"><span>(.+?)</span></div>', html)
links = re.findall(r'<div class="head"><img src="(.+?)"></div>', html)

metadata = []
change_res = ["冠顶苍隼"]
change_target = ["共鸣回响·冠顶苍隼"]
force = False

for (name, link) in zip(names, links):
    full_link = base_url + link
    print(full_link)
    img_name = os.path.basename(link)
    try:
        index = change_res.index(name)
        name = change_target[index]
    except ValueError:
        pass
    metadata.append({
        "name": name,
        "file": f"{img_name}"
    })
    if force == False and os.path.exists(img_name):
        continue
    img_data = requests.get(full_link).content
    with open(f"{img_name}", "wb") as img_file:
        img_file.write(img_data)

with open("metadata.json", "w", encoding="utf-8") as meta_file:
    json.dump(metadata, meta_file, ensure_ascii=False, indent=4)
    
print(names, links)