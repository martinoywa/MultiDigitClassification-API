from flask import Flask, jsonify

import requests
from PIL import Image
from io import BytesIO
import base64

from datetime import datetime

import os

app = Flask(__name__)


if "temp" not in os.listdir():
    os.mkdir("temp")


@app.route('/predict/<path:url>')
def predict(url):
    """
        Returns predicted label.
        parameter: 
            :url: link or image bytes
    """
    # generating filename
    datetimeObj = datetime.now()

    # if url if as https://miro.medi...
    if "https" in url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        file_name = "temp/" + datetimeObj.strftime("%d-%b-%Y--(%H-%M-%S)") + url[-4:]
        image.save(file_name)

        return jsonify({"url": file_name}), 200

    # if bytes for jpeg
    elif "data:image/jpeg;base64" in url:
        base_string = url.replace("data:image/jpeg;base64,", "")
        print(base_string[:10])
        decode = base64.b64decode(base_string)
        image = Image.open(BytesIO(decode))
        file_name = "temp/" + datetimeObj.strftime("%d-%b-%Y--(%H-%M-%S)") + ".jpg"
        image.save(file_name)

        return jsonify({"url": file_name}), 200

    # if bytes for png
    elif "data:image/png;base64" in url:
        base_string = url.replace("data:image/png;base64,", "")
        decode = base64.b64decode(base_string)
        image = Image.open(BytesIO(decode))
        file_name = "temp/" + datetimeObj.strftime("%d-%b-%Y--(%H-%M-%S)") + ".png"
        image.save(file_name)

        return jsonify({"url": file_name}), 200

    # error
    else:
        return jsonify({"Error": "Format not supported: Expecting link to image or image bytes"})


if __name__ == "__main__":
    app.run(debug=True)


# iVBORw0KGgoAAAAN
# iVBORw0KGgoAAAAN

# /9j/4AAQSkZJRgABAQAAAQABAAD