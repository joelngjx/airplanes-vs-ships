import gradio as gr
from PIL import Image
import torchvision as tv
import torch
from torch import nn
import torch.nn.functional as func
from torchvision import transforms

## Load models
modelResNetV0 = tv.models.resnet18(weights=None)
modelResNetV0.fc = nn.Linear(modelResNetV0.fc.in_features, 2)
modelResNetV0.load_state_dict(torch.load("models/cifarResNetV0.pt", map_location="cpu"))
modelResNetV0.eval()

modelResNetV1 = tv.models.resnet50(weights=None)
modelResNetV1.fc = nn.Linear(modelResNetV1.fc.in_features, 2)
modelResNetV1.load_state_dict(torch.load("models/cifarResNetV1.pt", map_location="cpu"))
modelResNetV1.eval()

class_names = ["Airplanes", "Ships"]

## Preprocessing data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## Making predictions
def classify(image, model_choice):
    if model_choice is None:
        return {}, "Please select a model before submitting."

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Model selections
    with torch.inference_mode():
        if model_choice == "ResNet18 (V0)":
            logits = modelResNetV0(input_tensor)
        else:
            logits = modelResNetV1(input_tensor)

        probas = func.softmax(logits, dim=1).squeeze().tolist()
        max_prob = max(probas)
        pred_index = torch.argmax(logits)

    # Probability threshold for class prediction -> 70%
    if max_prob < 0.7:
        return {class_names[i]: round(probas[i],2) for i in range(2)}, "It's probably not an airplane or a ship"

    return {class_names[i]: round(probas[i],2) for i in range(2)}, class_names[pred_index]

# Gradio GUI
interface = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(choices=["ResNet18 (V0)", "ResNet50 (V1)"], label="Choose Model")
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Predictions"),
        gr.Textbox(label="Predicted Class")
    ],
    title="Airplanes ✈️ VS Ships 🚢",
    description="Upload an image and classify it as an Airplane or a Ship using ResNet18 or ResNet50. The minimum confidence threshold for classifying an image as a class is 70%."
)

interface.launch(share=True)