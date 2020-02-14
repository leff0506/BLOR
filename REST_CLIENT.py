import os

import requests
import BASE64 as base
import json

bs = []
directory = "./test/samples"
i = 0
for filename in os.listdir(directory):
    if i == 2:
        break
    bs.append(base.file_to_base(directory + "/" + filename))
    i += 1

#
data = {"images": bs}
json_data = json.dumps(data)
headers = {'Content-type': 'application/json'}
res = requests.post('http://localhost:5000/todo/api/v1.0', json=json_data, headers=headers)
if res.ok:
    with open("Output1.json", "w") as text_file:
        text_file.write(json.dumps(res.json(), indent=4))
    print(json.dumps(res.json(), indent=4))
    # data = res.json()
    # i=0
    # for image_info in data['images']:
    #     image = image_info['body']
    #     base.base_to_file(image,"./test/REST/"+str(i)+".png")
    #     print(image_info['angle'])
    #     i+=1
