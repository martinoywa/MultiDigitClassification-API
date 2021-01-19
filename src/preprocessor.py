import io
from torchvision import transforms
from PIL import Image


def preprocess(image_bytes):
    """
    Returns transformed image tensor.
    :param image_bytes: image bytes from canvas.
    :return: transformed image.
    """
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.CenterCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # image_bytes are what we get from web request then grays the image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # sends a single image
    return transform(image).unsqueeze(0)
