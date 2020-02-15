import numpy as np
import BASE64 as base
import os
import DB as db
import claster as cl
import deskewing as desk
from collections import OrderedDict

def resp(image):
    result = OrderedDict()
    name = "./REST/got/" + str(np.random.randint(1e8)) + ".png"
    base.base_to_file(image, name)
    info = desk.normalize(name, name)
    body = base.file_to_base(name)

    info['body'] = body
    # info['body'] = "da"
    result["image"] = info

    clas = cl.self_recorded(name)
    os.remove(name)
    blocks = cl.get_blocks(clas[0], clas[1])
    rects = blocks[1]
    texts = blocks[0]
    union_rects = blocks[2]
    areas = []
    data = []
    for i in range(len(rects)):
        cur_area = {}
        coordinates = [rects[i].p1, rects[i].p2, rects[i].p3, rects[i].p4]
        cur_area["text"] = sort_text(union_rects[i])
        cur_area["coordinates"] = coordinates
        cur_area["description"] = str(i)

        areas.append(cur_area)

    # result["areas"] = "here will ba areas"
    result["areas"] = areas
    result["containers"] = "here must be array of containers"
    result["words"] = clas[2]
    # result["words"] = "response from google"
    paragraphs = []
    for i in range(len(rects)):
        cur_par = {}
        coordinates = [rects[i].p1, rects[i].p2, rects[i].p3, rects[i].p4]
        cur_par["text"] = sort_text(union_rects[i])
        cur_par["coordinates"] = coordinates
        paragraphs.append(cur_par)
    result["paragraphs"] = paragraphs
    # result["paragraphs"] = "here must be paragraphs"
    return result


def sort_text(rects):
    labels = cl.result_dfs(rects, db.WIDTH_FOR_SENTENCES, db.HEIGHT_FOR_SENTENCES)
    sentences_rects = []
    for i in range(len(np.unique(labels))):
        temp = []
        for j in range(len(rects)):
            if labels[j] == i:
                temp.append(rects[j])
        sentences_rects.append(temp)
    sentences_rects = sorted(sentences_rects, key=lambda a: a[0].center["y"])
    for i in range(len(sentences_rects)):
        sentences_rects[i] = sorted(sentences_rects[i], key=lambda a: a.center["x"])
    answer = ""

    for sent in sentences_rects:
        for word in sent:
            answer += word.word + " "
        answer += "\n"
    return answer
