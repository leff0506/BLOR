import json
from flask import Flask, jsonify, request
import response as resp
from collections import OrderedDict
app = Flask(__name__)
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
