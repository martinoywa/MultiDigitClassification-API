# MultiDigitClassification-API
API for Mutiple Digit Image Classification


# How to use API
```
Link-based images:
http://127.0.0.1:5000/api/v1/predict/{url_to _image}
---
Example:     
http://127.0.0.1:5000/api/v1/predict/https://www.ourcornermarket.com/assets/images/GU/numbers/standard-4.jpg
```

```
Images as bytes:
http://127.0.0.1:5000/api/v1/predict/{image_bytes}

---
Example:
http://127.0.0.1:5000/api/v1/predict/data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAAGQCAIAAAAP3aGbAAAXDUlEQVR4nO3dfXRU9Z3H8V+nkwcSQhggAUKJMcSEh4bIQ4nlQdEeVsG6uqW2XS2spx5ztLrqsdajFkt3xbMtq11k8WHpUWo56rosWksrtKyoBdE...
```

Supports both PNG and JPEG.

```
Output:
JSON Object
---
Example:
{
    "prediction": 1000
}
```

# Credits
API Developer: [Martin Oywa](https://www.martinoywa.engineer/)

Model Trainer: [Potter Hsu](https://github.com/potterhsu/SVHNClassifier-PyTorch)