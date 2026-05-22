# Plant Disease Detector

A deep learning web app that identifies plant diseases from leaf photos. Upload an image and get an instant prediction with confidence scores across 38 disease classes.

## Demo

Upload a leaf photo → model predicts the disease → top 5 confidence scores displayed.

## Model

Built with TensorFlow and EfficientNetB0 pretrained on ImageNet, fine-tuned on the PlantVillage dataset.

| | Accuracy |
|---|---|
| Simple CNN (baseline) | 88.0% |
| EfficientNetB0 frozen | 96.7% |
| EfficientNetB0 fine-tuned | 97.5% |

Training was done in two phases — first with EfficientNet frozen (only the top layers trained), then with the last 20 layers unfrozen at a learning rate of `1e-5`.

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 54,306 images across 38 plant disease classes covering 14 crop species including tomato, potato, apple, grape, and corn.

## Classes

The model recognizes 38 classes:

Apple scab, Apple black rot, Apple cedar rust, Apple healthy, Blueberry healthy, Cherry powdery mildew, Cherry healthy, Corn gray leaf spot, Corn common rust, Corn northern leaf blight, Corn healthy, Grape black rot, Grape esca, Grape leaf blight, Grape healthy, Orange citrus greening, Peach bacterial spot, Peach healthy, Pepper bacterial spot, Pepper healthy, Potato early blight, Potato late blight, Potato healthy, Raspberry healthy, Soybean healthy, Squash powdery mildew, Strawberry leaf scorch, Strawberry healthy, Tomato bacterial spot, Tomato early blight, Tomato late blight, Tomato leaf mold, Tomato septoria leaf spot, Tomato spider mites, Tomato target spot, Tomato yellow leaf curl virus, Tomato mosaic virus, Tomato healthy.

## Project Structure

```
plant-disease-detector/
├── app.py                    # Streamlit web app
├── plant_disease_model.h5    # trained model (download from Colab)
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

**1 — Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2 — Add the model file**

Download `plant_disease_model.h5` from Google Colab and place it in the same folder as `app.py`.

**3 — Run the app**

```bash
streamlit run app.py
```

## Requirements

```
streamlit
tensorflow
numpy
pillow
```

## Important — preprocessing

The model was trained without `rescale=1./255` because EfficientNetB0 handles its own internal preprocessing. The app reflects this — raw pixel values (0–255) are fed directly to the model. Do not normalize the input or predictions will be wrong.

## Training

Training was done in Google Colab with the following setup:

- Image size: 224×224
- Batch size: 32
- Phase 1: 5 epochs, `lr=1e-3`, EfficientNet frozen
- Phase 2: 5 epochs, `lr=1e-5`, last 20 layers unfrozen
- Augmentation: horizontal flip, rotation 15°, zoom 0.1
- Callbacks: EarlyStopping (patience 5), ReduceLROnPlateau (patience 3)

## Reproducibility

All random seeds are fixed to 0:

```python
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = '0'
```
