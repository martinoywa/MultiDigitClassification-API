from pathlib import Path


from src.loader import model_loader
from src.preprocessor import preprocess

checkpoint = Path("src/checkpoints/model-54000.pth")
model = model_loader(checkpoint)


def prediction(image_bytes):
    """
    Returns model prediction for drawing.
    :param image_bytes: image bytes from canvas.
    :return: predicted label.
    """
    tensor = preprocess(image_bytes)
    length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.forward(tensor)
    length_prediction = length_logits.max(1)[1]
    digit1_prediction = digit1_logits.max(1)[1]
    digit2_prediction = digit2_logits.max(1)[1]
    digit3_prediction = digit3_logits.max(1)[1]
    digit4_prediction = digit4_logits.max(1)[1]
    digit5_prediction = digit5_logits.max(1)[1]

    pred = [digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(),
            digit4_prediction.item(), digit5_prediction.item()]

    return int("".join(map(str, pred[:length_prediction.item()])))
