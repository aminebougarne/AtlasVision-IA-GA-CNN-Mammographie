# AtlasVision AI — Streamlit Demo

## Overview
This folder contains a deployable Streamlit prototype for **AtlasVision AI**, a GA-optimized CNN pipeline for explainable mammography classification.

## Features
- Mammography image upload
- Demonstrative AI prediction interface
- Simulated Grad-CAM visualization
- Genetic Algorithm optimization summary
- Evaluation metrics section

## Installation

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Deployment
You can deploy this app using **Streamlit Community Cloud** by connecting the GitHub repository and selecting `app.py` as the main file.

## Important note
The current prediction and Grad-CAM modules are demonstrative. They should be replaced by real PyTorch model inference after training.