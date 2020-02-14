import json
from flask import Flask, jsonify, request
import response as resp
from collections import OrderedDict
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
    data = json.loads(request.get_json())
    res = []
    for image in data["images"]:
        temp = resp.resp(image)
        res.append(temp)
    answer = OrderedDict()
    answer["status"] = "success"
    answer["data"] = res

    return jsonify(answer)


if __name__ == '__main__':
    app.config['JSON_SORT_KEYS'] = False
    app.run(debug=True)
