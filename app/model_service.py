
import torch
from transformers import pipeline


def load_model():
    # 0: 첫 번째 GPU, -1: CPU 사용
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "text-classification",
        model="WhitePeak/bert-base-cased-Korean-sentiment",
        model_kwargs={"weights_only": False},
        device=device
    )

def predict(model, text: str) -> dict:
    raw_result = model(text)[0]

    label_map = {"LABEL_1": "긍정", "LABEL_0": "부정"}
    display_label = label_map.get(raw_result["label"], raw_result["label"])

    return {
        "label": display_label,
        "confidence": round(float(raw_result["score"]), 4)
    }
