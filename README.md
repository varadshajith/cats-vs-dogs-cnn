# Cats vs Dogs CNN Classifier

A convolutional neural network implementation for binary image classification, achieving 86.1% validation accuracy on the Kaggle Cats and Dogs dataset.

## Overview

This project uses TensorFlow/Keras to build a custom CNN architecture that classifies images as either cats or dogs. The model was trained on 20,000 images using data augmentation and dropout regularization to prevent overfitting.

## Model Architecture

```
Input (150x150x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (512) + ReLU + Dropout(0.5)
    ↓
Dense (1) + Sigmoid
```

Input: 150x150x3 RGB images
Output: Binary classification (0 = cat, 1 = dog)

## Performance

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 86.5% | 86.1% |
| Loss | 0.313 | 0.324 |
| Epochs | 10 | 10 |
| Training Time | ~24 minutes | GPU (T4) |

## Dataset

25,000 images from Kaggle's Dogs vs Cats dataset
- Training: 20,000 images (80%)
- Validation: 5,000 images (20%)
- Preprocessing: Resized to 150x150, normalized to [0,1]

Data augmentation applied during training:
- Rotation (40°)
- Width/height shift (20%)
- Shear transformation (20%)
- Zoom (20%)
- Horizontal flip

## Technical Stack

- TensorFlow 
- Keras 
- NumPy
- Matplotlib
- Seaborn
- Google Colab (GPU)

## Installation

```bash
git clone https://github.com/varadshajith/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn
pip install -r requirements.txt
```

## Usage

### Training

Open `Cat_dog_CNN_completed.ipynb` in Google Colab. The notebook includes:
- Data loading and preprocessing
- Model architecture definition
- Training with callbacks
- Performance visualization

### Prediction

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('cat_dog_classifier.h5')

img = image.load_img('test.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
print("Dog" if prediction > 0.5 else "Cat")
```

## Training Details

The model uses binary cross-entropy loss and Adam optimizer. Training progresses:

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 59.8% | 67.4% | 0.665 | 0.626 |
| 5 | 80.8% | 80.1% | 0.422 | 0.444 |
| 10 | 86.5% | 86.1% | 0.313 | 0.324 |

Dropout at 0.5 rate prevents overfitting. Validation loss stabilizes around epoch 8-9.

## Project Structure

```
cats-vs-dogs-cnn/
├── Cat_dog_CNN_completed.ipynb
├── README.md
└── requirements.txt
```

## License

MIT License

## Contact

Varad Shajith
- GitHub: [@varadshajith](https://github.com/varadshajith)
- Email: varadshajith@gmail.com
