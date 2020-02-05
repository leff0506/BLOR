import requests
import json
import numpy as np
import base64
import deskewing as desk
def detect_text(origin):
    url = 'https://vision.googleapis.com/v1/images:annotate?alt=json&key=AIzaSyAwy6okZ-wrLFCehajOsN8S9fn_4d3eoWI'
    with open(origin, "rb") as img_file:
        base= base64.b64encode(img_file.read())
        base = base.decode('utf-8')
    req = '{"requests": [{"image": {"content":"'
    req += base
    req +='"},"features": [{"type": "TEXT_DETECTION","model": "builtin/latest"}],"imageContext": {"languageHints": "en"}}]}'
    headers = {'Content-type': 'application/json',
               'Accept': 'text/plain',}
    answer = requests.post(url, data=req, headers=headers)
    answer = answer.json()
    return answer


class rect(object):
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        #         print(p1,p2,p3,p4)
        self.center = {}
        temp = 0
        tempx = 0
        if 'x' in p1.keys():
            temp += 1
            tempx += p1['x']
        if 'x' in p2.keys():
            temp += 1
            tempx += p2['x']
        if 'x' in p3.keys():
            temp += 1
            tempx += p3['x']
        if 'x' in p4.keys():
            temp += 1
            tempx += p4['x']

        self.center['x'] = tempx / temp;
        temp = 0
        tempy = 0
        if 'y' in p1.keys():
            temp += 1
            tempy += p1['y']
        if 'y' in p2.keys():
            temp += 1
            tempy += p2['y']
        if 'y' in p3.keys():
            temp += 1
            tempy += p3['y']
        if 'y' in p4.keys():
            temp += 1
            tempy += p4['y']

        self.center['y'] = tempy / temp;

    def set_word(self, word):
        self.word = word

    def __str__(self):
        res = ""
        res += str(self.p1)
        res += " "
        res += str(self.p2)
        res += " "
        res += str(self.p3)
        res += " "
        res += str(self.p4)
        res += " "
        res += "center : "
        res += str(self.center)
        res += " word : "
        res += self.word
        return res


def json_to_rects(js):
    data = js
    data = data['responses'][0]['textAnnotations']
    rects = []
    for i in range(1, len(data)):
        temp = []
        for point in data[i]['boundingPoly']['vertices']:
            temp.append(point)
        rects.append(rect(temp[0], temp[1], temp[2], temp[3]))
        rects[i - 1].set_word(data[i]['description'])
    return rects
def dict_to_np(rects):
    X = [[0 for i in range(2)] for j in range(len(rects))]
    for i in range(len(rects)):
        X[i][0] = int(rects[i].center['x'])
        X[i][1] = int(rects[i].center['y'])
    X = np.array(X,dtype = int)
    return X


def filt(rects):
    import re
    h = 0
    maersk = "\d{9,10}"
    AAAChina = "[A-Z]{3}\d{8}"
    ZIM = "[A-Z]{7,8}\d{7,8}"
    SeaGo = "[A-Z]{3}\d{6}"
    CM_CGM = "[A-Z]{3}\d{7}"
    Pearl = "[A-Z]{2}\d{8}"
    # for r in rects:
    #     if re.fullmatch(maersk, r.word) or re.fullmatch(AAAChina, r.word) or re.fullmatch(ZIM, r.word) or re.fullmatch(SeaGo, r.word) or re.fullmatch(CM_CGM, r.word) or re.fullmatch(Pearl, r.word):
    #         h = r.p3['y'] - r.p1['y']
    #         print(r.word, h)
    #         break
    h = 13
    res = []
    eps = 0
    forbiden = ["consignee", "vessel", "voyage", "no."]
    # forbiden = []
    for r in rects:
        if r.p3.get('y',0) - r.p1.get('y',1e7) >= h - eps and 'x' in r.p1 and 'y' in r.p1 and 'x' in r.p2 and 'y' in r.p2 and 'x' in r.p3 and 'y' in r.p3 and 'x' in r.p4 and 'y' in r.p4 and not r.word.lower() in forbiden:
            res.append(r)

    return res


def result_claster(X, labels, photo, rects, save='claster_res.png'):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    col = [[0, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [1, 0, 1, 0.5], [1, 1, 0, 0.5], [0, 1, 1, 0.5],
           [1, 0, 0, 0.1], [0.5, 0.5, 0, 0.5]]
    #     col = [[0,0,0,0.5],[0,1,0,0.0],[0,0,1,0],[1,0,1,0],[1,1,0,0],[0,1,1,0],[1,0,0,0],[0.5,0.5,0,0]]
    n = len(np.unique(labels))
    colors = len(X) * [0]
    for i in range(len(X)):
        colors[i] = col[labels[i] % len(col)]
    #     for i in range(len(X)):
    #         if labels[i] == 0 :
    #             colors[i] = col[0]
    #         else:
    #             colors[i] = col[1]
    plt.figure(figsize=(30, 40))
    image = plt.imread(photo)
    plt.imshow(image)
    plt.scatter(X[:, 0], X[:, 1], color=colors, s=400)

    temp = []
    for i in range(n):
        x1 = 1e7
        x2 = 0
        y1 = 1e7
        y2 = 0
        for j in range(len(rects)):
            if labels[j] == i:
                if 'x' in rects[j].p1.keys():
                    x1 = min(x1, rects[j].p1['x'])
                if 'x' in rects[j].p2.keys():
                    x2 = max(x2, rects[j].p2['x'])

                if 'y' in rects[j].p1.keys():
                    y1 = min(y1, rects[j].p1['y'])
                if 'y' in rects[j].p3.keys():
                    y2 = max(y2, rects[j].p3['y'])

        temp.append(rect({'x': x1, 'y': y1}, {'x': x2, 'y': y1}, {'x': x2, 'y': y2}, {'x': x1, 'y': y2}))
        plt.gca().add_patch(patches.Rectangle((temp[i].p1['x'], temp[i].p1['y']), temp[i].p2['x'] - temp[i].p1['x'],
                                              temp[i].p3['y'] - temp[i].p1['y'], linewidth=1, edgecolor='none',
                                              facecolor=col[i % len(col)]))
    plt.savefig(save)
    # plt.show()
def dist_rect(a ,b):
    w = 1e7
    h = 1e7
    if (a.p1['x'] <= b.p1['x'] <= a.p3['x']) or (a.p1['x'] <= b.p3['x'] <= a.p3['x']) or (
            b.p1['x'] <= a.p1['x'] <= b.p3['x']) or (b.p1['x'] <= a.p3['x'] <= b.p3['x']):
        w = 0
    else:
        w = min(w,abs(a.p1['x'] - b.p1['x']))
        w = min(w,abs(a.p1['x'] - b.p3['x']))
        w = min(w,abs(a.p3['x'] - b.p3['x']))
        w = min(w,abs(a.p3['x'] - b.p1['x']))
        w = max(w,0)
    if (a.p1['y'] <= b.p1['y'] <= a.p3['y']) or (a.p1['y'] <= b.p3['y'] <= a.p3['y']) or (
            b.p1['y'] <= a.p1['y'] <= b.p3['y']) or (b.p1['y'] <= a.p3['y'] <= b.p3['y']):
        h = 0
    else:
        h = min(h,abs(a.p1['y'] - b.p1['y']))
        h = min(h,abs(a.p1['y'] - b.p3['y']))
        h = min(h,abs(a.p3['y'] - b.p3['y']))
        h = min(h,abs(a.p3['y'] - b.p1['y']))
        h = max(h,0)
    if w== 0 and h == 0:
        w=h=1
    return (w,h)


def result_dfs(rects, width, height, lines=[]):
    from collections import defaultdict
    n = len(rects)

    def collision_one(a, b, line):
        (w, h) = dist_rect(a, b)
        if w == 0:
            return False
        if ((line[1] <= a.center['y'] <= line[2]) or (line[1] <= b.center['y'] <= line[2])) and min(a.center['x'], b.center['x']) <= line[0] and max(a.center['x'], b.center['x']) >= line[0]:
            return True
        elif ((line[1] <= a.p1['y'] <= line[2]) or (line[1] <= b.p1['y'] <= line[2])) and min(a.p1['x'], b.p1['x']) <= line[0] and max(a.p1['x'], b.p1['x']) >= line[0]:
            return True
        elif ((line[1] <= a.p3['y'] <= line[2]) or (line[1] <= b.p3['y'] <= line[2])) and min(a.p3['x'], b.p3['x']) <= line[0] and max(a.p3['x'], b.p3['x']) >= line[0]:
            return True
        elif ((line[1] <= a.p3['y'] <= line[2]) or (line[1] <= b.p1['y'] <= line[2])) and min(a.p3['x'], b.p1['x']) <= line[0] and max(a.p3['x'], b.p1['x']) >= line[0]:
            return True
        elif ((line[1] <= a.p1['y'] <= line[2]) or (line[1] <= b.p3['y'] <= line[2])) and min(a.p1['x'], b.p3['x']) <= line[0] and max(a.p1['x'], b.p3['x']) >= line[0]:
            return True
        else:
            return False

    def collision(a, b, lines):
        for line in lines:
            if collision_one(a, b, line):
                return True
        return False

    class Graph:
        def __init__(self, n):
            self.graph = [[i] for i in range(n)]

        def addEdge(self, u, v):
            self.graph[u].append(v)

        def DFSUtil(self, v, visited, cur):
            self.labels[v] = cur
            visited[v] = True
            for i in self.graph[v]:
                if visited[i] == False:
                    self.DFSUtil(i, visited, cur)

        def DFS(self):
            visited = [False] * (len(self.graph))
            # print(len(visited))
            self.labels = [0] * (len(self.graph))
            cur = 0
            #             self.DFSUtil(0, visited,cur)
            for i in range(len(self.graph)):
                if visited[i] == False:
                    self.DFSUtil(i, visited, cur)
                    cur += 1

    #     for i in range(n):
    #         print(rects[i])
    g = Graph(n)
    for i in range(n):
        for j in range(n):
            if not i == j:
                (w, h) = dist_rect(rects[i], rects[j])

                if w <= width and h <= height and not collision(rects[i], rects[j], lines):
                    g.addEdge(i, j)
    #                     print(i,j,w,h)
    # print(g.graph[11])
    #     print(rects[38])
    #     print(rects[39])
    #     print(rects[40])
    #     print(rects[41])
    #     print(g.graph[0])
    g.DFS()

    #     print(rects[4])
    #     print(rects[5])
    return g.labels


def get_vert_lines(rects, labels):
    points = []
    for i in range(len(np.unique(labels))):
        x = 1e7
        y1 = 1e7
        y2 = 1e7
        for j in range(len(rects)):
            if labels[j] == i:
                x = min(x, rects[j].p1['x'])
                y1 = min(y1, rects[j].p1['y'])
                y2 = min(y2, rects[j].p3['y'])
        points.append([x, y1])
        points.append([x, y2])
    points = sorted(points)
    #     for p in points:
    #         print(p[0],p[1])
    x_eps = 10
    y_eps = 50
    result = []
    was = []
    was.append(1)
    was_treshhold = 3
    j = 0
    result.append([points[0][0], points[0][1], points[0][1]])
    for i in range(1, len(points)):
        done = False
        for k in range(len(result)):
            if abs(points[i][0] - result[k][0]) <= x_eps and abs(points[i][1] - result[k][2]) <= y_eps:
                done = True
                was[k] += 1
                result[k][2] = max(result[k][2], points[i][1])
                break
        if not done:
            was.append(1)
            result.append([points[i][0], points[i][1], points[i][1]])
    answer = []
    y1_eps = 5
    y2_eps = 10
    for i in range(len(result)):
        if was[i] >= was_treshhold:
            result[i][1] = max(0, result[i][1] - y1_eps)
            result[i][2] += y2_eps
            answer.append(result[i])
    return answer
def self_recorded(origin , dest):
    print("start :",origin)
    rects = filt(json_to_rects(detect_text(origin)))
    X = dict_to_np(rects)
    labels = result_dfs(rects,20,0)
    lines = get_vert_lines(rects,labels)
    # lines = []
    # draw_lines(origin,lines)
#     print(type(rects))
    #25 15
    result_claster(X,result_dfs(rects,25,15,lines),origin,rects,save = dest)
    print("end :", origin)
def draw_lines(photo,lines):
    import matplotlib.pyplot as plt
    col = [[0,0,0,0.5],[0,1,0,0.5],[0,0,1,0.5],[1,0,1,0.5],[1,1,0,0.5],[0,1,1,0.5],[1,0,0,0.1],[0.5,0.5,0,0.5]]
    plt.figure(figsize=(30,40))
    image = plt.imread(photo)
    plt.imshow(image)
    for i in range(len(lines)):
        x_values = 2*[lines[i][0]]
        y_values = [lines[i][1],lines[i][2]]
        plt.plot(x_values,y_values,c = col[i%len(col)],linewidth=8)
    plt.show()
import os
# directory = './test/samples'
# dest = "./test/norm/"
# for filename in os.listdir(directory):
#     desk.normalize(directory+"/"+filename,dest+filename)
directory = "./test/norm"
dest = "./test/self_recorded/"
for filename in os.listdir(directory):
    self_recorded(directory+"/"+filename,dest+filename)
# rects = filt(json_to_rects(detect_text("./test/norm/data.png")))
# for rect in rects:
#     print(rect)
# print(rects[5],rects[11])
# print(dist_rect(rects[5],rects[11]))
# result_dfs(rects,20, 0)
# self_recorded("./test/norm/data.png","./test/self_recorded/data.png")