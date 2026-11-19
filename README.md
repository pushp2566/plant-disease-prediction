# Plant Disease Prediction — CNN Image Classifier

A convolutional neural network that identifies plant diseases from leaf photographs, trained on the
PlantVillage dataset and served through a Streamlit web app.

**38 disease and healthy classes · 14 crop species · 88.28% validation accuracy**

---

## What it does

Upload a photo of a plant leaf and the model predicts which of 38 categories it belongs to — for
example `Tomato___Late_blight`, `Apple___Black_rot`, or `Blueberry___healthy`. Sample images are
included in `test_images/` so the app can be tried immediately.

## Dataset

[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — leaf images
labelled by crop and disease.

| Property | Value |
|---|---|
| Classes | 38 |
| Crop species | 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato) |
| Input resolution | 224 × 224 × 3 |
| Train/validation split | 80 / 20 |
| Normalization | Pixel values rescaled to [0, 1] |

## Model architecture

```
Input (224, 224, 3)
  Conv2D(32, 3×3, relu)  →  MaxPooling2D(2×2)
  Conv2D(64, 3×3, relu)  →  MaxPooling2D(2×2)
  Flatten
  Dense(256, relu)
  Dense(38, softmax)
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Epochs | 5 |
| Batch size | 32 |
| Data loader | `ImageDataGenerator` (rescale + validation split) |

## Results

| Metric | Value |
|---|---|
| Training accuracy | 97.60% |
| **Validation accuracy** | **88.28%** |
| Validation loss | 0.5091 |

**A note on the gap.** Training accuracy sits about 9 points above validation accuracy, so the model
is overfitting. This is expected given the architecture: only two convolutional blocks, no dropout,
no batch normalization, and no augmentation beyond rescaling. The most direct improvements would be
adding dropout before the dense layer, real augmentation (rotation, flips, zoom, brightness), and
either early stopping or more epochs with regularization. Transfer learning from a pretrained
backbone such as MobileNetV2 or EfficientNet would likely push validation accuracy well past 95%
with less training time.

## Project structure

```
app/
  main.py                  Streamlit app — upload, preprocess, predict
  class_indices.json       Index → class-name mapping (38 entries)
  requirements.txt         Python dependencies
  Dockerfile               Container build
  config.toml              Streamlit server config
  trained_model_link.txt   Download link for the trained .h5 weights
model_training_notebook/
  Plant_Disease_Prediction_CNN_Image_Classifier.ipynb    Full training pipeline
test_images/               Sample leaves for trying the app
```

## Running locally

The trained weights are **not** stored in this repository because of their size. Download them first,
or the app will fail to start.

```bash
git clone https://github.com/pushp2566/plant-disease-prediction
cd plant-disease-prediction/app

# 1. Download the trained model using the link in trained_model_link.txt
#    and place it at exactly this path:
#    app/trained_model/plant_disease_prediction_model.h5
mkdir -p trained_model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run main.py
```

Then open http://localhost:8501 and upload any image from `test_images/`.

### With Docker

```bash
cd app
docker build -t plant-disease .
docker run -p 80:80 plant-disease
```

## Retraining

Open `model_training_notebook/Plant_Disease_Prediction_CNN_Image_Classifier.ipynb`, point the data
directory at your local copy of the PlantVillage dataset, and run the cells in order. The notebook
covers loading, preprocessing, model definition, training, evaluation, and saving the `.h5` weights.

## Built with

Python · TensorFlow · Keras · NumPy · Pillow · Streamlit · Docker

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Dataset courtesy of the [PlantVillage](https://plantvillage.psu.edu/) project.
