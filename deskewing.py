import numpy as np
import cv2
import random
import json
import base64
import requests


def deskewing(origin, to):
    image = cv2.imread(origin)
    shift = 75
    (h, w) = image.shape[:2]
    temp = np.ones((h + 2 * shift, w + 2 * shift, 3), dtype='uint8')
    temp[:, :, :] = 255

    temp[shift:h + shift, shift:w + shift, :] = image
    image = temp
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    res = cv2.minAreaRect(coords)
    angle = res[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    dimensions = gray.shape
    h = gray.shape[0]
    w = gray.shape[1]
    x1 = 0
    while min(gray[:, x1]) == 255:
        x1 += 1
    y1 = 0
    while min(gray[y1, :]) == 255:
        y1 += 1
    x2 = w - 1
    while min(gray[:, x2]) == 255:
        x2 -= 1
    y2 = h - 1
    while min(gray[y2, :]) == 255:
        y2 -= 1
    rotated = rotated[y1:y2 + 1, x1:x2 + 1]
    res_w = 1600
    res_h = 2300
    rotated = cv2.resize(rotated, (res_w, res_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(to, rotated)
def read_json(origin):
    with open(origin) as json_file:
        data = json.load(json_file)
    return data
def crop(origin,origin_data):
    data = origin_data
    data = data['responses'][0]['textAnnotations']
    x1 = 1e5
    y1 = 1e5
    x2 = 0
    y2 = 0
    for i in range(len(data)):
        for point in data[i]['boundingPoly']['vertices']:
            x1 = min(x1,point.get('x',1e7))
            x2 = max(x2,point.get('x',0))
            y1 = min(y1,point.get('y',1e7))
            y2 = max(y2,point.get('y',0))
    shift = 5
    image = cv2.imread(origin)
    (h, w) = image.shape[:2]
    x1 = max(0,x1-shift)
    x2 = min(w,x2+shift)
    y1 = max(0,y1-shift)
    y2 = min(h,y2+shift)
    image = image[y1:y2+1,x1:x2+1]
    res_w = 1600
    res_h = 2300
    cv2.imwrite(origin,image)
    deskewing(origin,origin)
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
def normalize(origin, dest):
    deskewing(origin,dest)
    crop(dest,detect_text(dest))
    deskewing(dest,dest)
