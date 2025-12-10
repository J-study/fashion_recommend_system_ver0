import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

clf_model = None   # ←追加（例外防止）
clf_tf = None      # ←追加

def init_classifier(model_obj, transform):
    global clf_model, clf_tf
    clf_model = model_obj
    clf_model.eval()  # 評価モードにする
    clf_tf = transform

def score_image_pil(img: Image.Image) -> float:
    if clf_model is None or clf_tf is None:
        raise ValueError("Classifier not initialized. Call init_classifier() first.")

    x = clf_tf(img).unsqueeze(0).to(next(clf_model.parameters()).device)

    with torch.no_grad():
        logits = clf_model(x)
        prob = F.softmax(logits, dim=1)[0]  # 2クラス想定（0=bad, 1=good）

    return float(prob[1].cpu().item())
