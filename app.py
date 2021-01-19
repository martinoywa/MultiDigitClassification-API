from flask import Flask, jsonify

import requests
import base64

from datetime import datetime

from src.inference import prediction


app = Flask(__name__)


@app.route('/api/v1/predict/<path:url>')
def predict(url):
    """
        Returns predicted label.
        params: link or image bytes
    """
    # generating filename
    datetimeObj = datetime.now()

    # if url if as https://miro.medi...
    if "https" in url:
        response = requests.get(url)
        pred = prediction(response.content)

        return jsonify({"url": pred}), 200


    # if bytes for jpeg
    elif "data:image/jpeg;base64" in url:
        base_string = url.replace("data:image/jpeg;base64,", "")
        decode = base64.b64decode(base_string)
        pred = prediction(decode)

        return jsonify({"url": pred}), 200


    # if bytes for png
    elif "data:image/png;base64" in url:
        base_string = url.replace("data:image/png;base64,", "")
        decode = base64.b64decode(base_string)
        pred = prediction(decode)

        return jsonify({"url": pred}), 200

    # error
    else:
        return jsonify({"Error": "Format not supported: Expecting link to image or image bytes"})


if __name__ == "__main__":
    app.run(debug=True)
