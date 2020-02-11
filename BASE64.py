import base64


def file_to_base(name):
    with open(name, "rb") as img_file:
        base = base64.b64encode(img_file.read())
        base = base.decode('utf-8')
    return base


def base_to_file(base, name):
    with open(name, "wb") as fh:
        fh.write(base64.decodebytes(base.encode('utf-8')))
