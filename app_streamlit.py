import streamlit as st
from PIL import Image
import torchvision as tv
import torch
from torch import nn
import torch.nn.functional as func


## Load models
modelResNetV0, modelResNetV1 = tv.models.resnet18(weights=None), tv.models.resnet50(weights=None)
modelResNetV0.fc = nn.Linear(modelResNetV0.fc.in_features, out_features=2)
modelResNetV0.load_state_dict(torch.load("models/cifarResNetV0.pt"))
modelResNetV0.eval()


modelResNetV1.fc = nn.Linear(modelResNetV1.fc.in_features, out_features=2)
modelResNetV1.load_state_dict(torch.load("models/cifarResNetV1.pt"))
modelResNetV1.eval()


## GUI
st.title("Airplanes ✈️ VS Ships 🚢")
class_names = ["Airplanes", "Ships"]

file_uploader = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if file_uploader is not None:
    image = Image.open(file_uploader).convert("RGB")
    st.image(file_uploader, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded!!")


    # Choose Model
    model_selection = st.radio("Choose a model: ", ("ResNet18 (V0)", "ResNet50 (V1)"), index=None) 

    # Model preprocessing
    transform = tv.transforms.Compose([        
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    input = transform(image).unsqueeze(dim=0)


    # Making predictions
    if st.button("Classify Image"):
        if model_selection is None:
            st.error("Please select a model before submitting.")
        else:
            with torch.inference_mode():
                if model_selection == "ResNet18 (V0)":
                    logits = modelResNetV0(input)
                else:
                    logits = modelResNetV1(input)

                probas = func.softmax(logits, dim=1).squeeze().tolist() 

        # Display results
        st.subheader("Predictions:")
        for i, prob in enumerate(probas):
            st.write(f"{class_names[i]}: {prob*100:.2f}%")

        # Classification proba threshold 70%
        if max(probas) >= 0.7:
            st.success(f"Predicted Class: **{class_names[torch.argmax(logits).item()]}**")
        else:
            st.warning("It's probably not an airplane or a ship.")