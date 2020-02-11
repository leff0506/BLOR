import base64
import os

import claster as cl
import deskewing as desk
import tempfile
import json
import numpy as np
from flask import Flask, jsonify, request
import BASE64 as base

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})


@app.route('/todo/api/v1.0', methods=['POST'])
def req():
    print(request.json)
    data = json.loads(request.get_json())
    res = []
    # name = np.random.randint(0,1e8)
    #
    # for image in data["images"]:
    #     print(image)
    #     name = str(np.random.randint())
    #     with open(name+".png", "wb") as fh:
    #         fh.write(base64.decodebytes(image.encode()))
    #     cl.self_recorded(name+".png",name+"_result")

    for image in data["images"]:
        print(image)
        name = "./test/REST/got/" + str(np.random.randint(1e8)) + ".png"
        base.base_to_file(image, name)
        info = desk.normalize(name, name)
        body = base.file_to_base(name)
        os.remove(name)
        info['body'] = body
        res.append(info)
    return jsonify({"images": res})


if __name__ == '__main__':
    app.run(debug=True)
