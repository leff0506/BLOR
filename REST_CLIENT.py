import os

import requests
import BASE64 as base
import json
bs = []
directory = "./test/samples"
for filename in os.listdir(directory):
    bs.append(base.file_to_base(directory+"/"+filename))

#
data  = {"images": bs}
json_data = json.dumps(data)
headers = {'Content-type': 'application/json'}
res = requests.post('http://localhost:5000/todo/api/v1.0', json=json_data,headers = headers)
if res.ok:
    data = res.json()
    i=0
    for image_info in data['images']:
        image = image_info['body']
        base.base_to_file(image,"./test/REST/"+str(i)+".png")
        print(image_info['angle'])
        i+=1
