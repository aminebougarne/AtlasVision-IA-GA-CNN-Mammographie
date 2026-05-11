import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(
    page_title="AtlasVision AI",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AtlasVision AI")
st.subheader("GA-Optimized CNN Pipeline for Explainable Mammography Classification")

st.markdown("""
**AtlasVision AI** is a medical computer vision prototype for mammography classification.  
It combines a CNN-based diagnostic pipeline with Genetic Algorithm optimization and explainability using Grad-CAM.

> This Streamlit application is a demonstrative prototype for hackathon presentation.  
> The prediction module can later be connected to a trained PyTorch model.
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Mammography Image")
    uploaded_file = st.file_uploader(
        "Upload a mammography image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Mammogram", use_container_width=True)
    else:
        st.info("Please upload a mammography image to start the analysis.")

with col2:
    st.header("🧠 AI Prediction")

    if uploaded_file is not None:
        classes = ["Normal", "Benign", "Malignant"]

        # Demo probabilities. Replace with real model inference later.
        probs = np.random.dirichlet(np.ones(3), size=1)[0]
        predicted_class = classes[int(np.argmax(probs))]

        st.metric("Predicted Class", predicted_class)

        for cls, prob in zip(classes, probs):
            st.progress(float(prob), text=f"{cls}: {prob:.2%}")

        if predicted_class == "Malignant":
            st.error("High-risk prediction. Clinical confirmation is required.")
        elif predicted_class == "Benign":
            st.warning("Benign lesion prediction. Further clinical assessment may be considered.")
        else:
            st.success("Normal prediction. Clinical validation remains necessary.")

st.divider()

if uploaded_file is not None:
    st.header("🔥 Explainability: Grad-CAM Visualization")

    image_array = np.array(image.resize((224, 224)))

    # Simulated Grad-CAM heatmap for demo purposes
    heatmap = np.random.rand(224, 224)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_array, cmap="gray")
    ax.imshow(heatmap, alpha=0.45)
    ax.axis("off")
    st.pyplot(fig)

    st.caption(
        "Grad-CAM demonstration: this heatmap is simulated. "
        "In the final model, it should be generated from the CNN feature maps."
    )

st.divider()

st.header("⚙️ Genetic Algorithm Optimization")

st.markdown("""
The Genetic Algorithm is designed to optimize CNN hyperparameters such as:

- Number of convolutional layers  
- Number of filters  
- Kernel size  
- Dropout rate  
- Learning rate  
- Batch size  

The best-performing individual is selected according to validation accuracy or F1-score.
""")

ga_table = {
    "Hyperparameter": [
        "Conv Layers",
        "Filters",
        "Kernel Size",
        "Dropout",
        "Learning Rate",
        "Batch Size"
    ],
    "Example Search Space": [
        "2–5",
        "16–128",
        "3×3 / 5×5",
        "0.2–0.5",
        "1e-5–1e-3",
        "8 / 16 / 32"
    ]
}

st.table(ga_table)

st.divider()

st.header("📊 Model Evaluation Metrics")

st.markdown("""
Planned evaluation metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  
- ROC-AUC curve  
""")

st.info("This interface is ready for deployment and can be connected to a trained PyTorch model.")

st.markdown("""
---
### 👤 Author  
**Amine Bougarne**  
AtlasVision AI — Evolutionary Intelligence for Breast Cancer Detection
""")