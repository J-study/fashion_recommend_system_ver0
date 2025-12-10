import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_wrappers import init_sd, generate_images
from classifier_wrapper import init_classifier, score_image_pil

from torchvision.models import efficientnet_b0

# 1) モデルロード
#Stable Diffusion Load
init_sd(
    lora_path="/content/drive/MyDrive/Loras/zemi_notXL/output",
    lora_weight_name="zemi_notXL-10.safetensors"
)

#Classifier Load
model = efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("/content/fashion_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

init_classifier(model, transform)

# 2) 推薦関数（UI が実行する部分）
def recommend(item_image, prompt_text):
    """服画像からSD生成→スコア付け→上位3枚返す"""

    # サイズ揃える
    init_img = item_image.convert("RGB").resize((512,512))

    # StableDiffusion 生成
    generated = generate_images(
        init_img,
        prompt=prompt_text or "A full-body fashion photo, realistic, stylish, neutral background",
        total_images=20,
        batch_size=5
    )

    # スコア付け
    scored = [(img, score_image_pil(img)) for img in generated]

    # ソート
    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = scored[:3]

    # 出力形式整形
    output_imgs = [t[0] for t in top3]
    scores = [f"Score: {t[1]:.4f}" for t in top3]

    return output_imgs, scores


# 3) Gradio UI 定義
demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Image(type="pil", label="Upload Fashion Item"),
        gr.Textbox(value="A stylish full-body outfit", label="Optional Prompt")
    ],
    outputs=[
        gr.Gallery(label="Best Recommended Outfits", columns=3, height="auto"),
        gr.Textbox(label="Scores")
    ],
    title="AI Fashion Outfit Recommender",
    description="Upload a clothing item to generate and score AI outfit combinations."
)


demo.launch(share=True)
