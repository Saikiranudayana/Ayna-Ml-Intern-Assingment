# UNet2D Conditional Model for Polygon Coloring

## 🎯 Project Overview

This project implements a **Conditional UNet2D** model for polygon coloring, where the model learns to color grayscale polygon images based on text-based color conditions. This is a conditional image generation task that demonstrates the power of deep learning in understanding both visual patterns and textual conditions.

**Assignment**: Ayna ML Intern Role

## 🏗️ Architecture

- **Model**: Conditional UNet2D with embedding layers
- **Input**: Grayscale polygon images (64x64) + color condition (text)
- **Output**: Colored polygon images (64x64, RGB)
- **Parameters**: 13.4M trainable parameters
- **Conditioning**: Text-to-embedding with spatial broadcasting

## 📊 Dataset

- **Training samples**: 56 images
- **Validation samples**: 5 images
- **Polygon types**: Circle, Square, Triangle, Hexagon, Pentagon, Octagon, Diamond, Star
- **Colors**: Blue, Cyan, Green, Magenta, Orange, Purple, Red, Yellow
- **Format**: PNG images, JSON metadata

## 🚀 Key Features

1. **Conditional Generation**: Colors polygons based on text conditions
2. **UNet Architecture**: Skip connections preserve spatial information
3. **Embedding System**: Converts text colors to numerical representations
4. **Progressive Training**: Learning rate scheduling and early stopping
5. **Model Persistence**: Automatic saving of best models
6. **Visualization**: Real-time training progress and results
7. **Inference Pipeline**: Easy-to-use prediction functions

## 📈 Performance

- **Best Validation Loss**: 0.1672 (MSE)
- **Training Epochs**: 55 (with early stopping)
- **Learning Rate**: 0.0001 → 0.00005 (adaptive)
- **Training Time**: ~3 minutes on CPU

## 🔧 Technical Implementation

### Model Architecture:
```
Conditional UNet2D:
├── Condition Embedding (8 colors → 64 dims)
├── Encoder Path:
│   ├── DoubleConv (3+64 → 64)
│   ├── Down (64 → 128)
│   ├── Down (128 → 256)
│   ├── Down (256 → 512)
│   └── Down (512 → 1024)
└── Decoder Path:
    ├── Up (1024 → 512) + skip
    ├── Up (512 → 256) + skip
    ├── Up (256 → 128) + skip
    ├── Up (128 → 64) + skip
    └── Output Conv (64 → 3)
```

### Training Details:
- **Loss Function**: MSE Loss
- **Optimizer**: Adam
- **Batch Size**: 16
- **Data Augmentation**: Normalization, Resizing
- **Regularization**: Batch Normalization

## 📁 Project Structure

```
├── main.ipynb              # Complete implementation
├── dataset/                # Training and validation data
│   ├── training/
│   │   ├── inputs/         # Grayscale polygon images
│   │   ├── outputs/        # Colored target images
│   │   └── data.json       # Metadata
│   └── validation/
├── models/                 # Saved model checkpoints
│   ├── best_model.pth     # Best model during training
│   └── final_model.pth    # Final model state
└── readme.md              # This documentation
```

## 🛠️ Requirements

```python
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
pillow>=8.0.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

## 🚀 Usage

### Training:
```python
# Run all cells in main.ipynb
# The notebook contains complete implementation from data loading to evaluation
```

### Inference:
```python
from main import predict_single_image, load_best_model

# Load trained model
model, color_map = load_best_model('models/best_model.pth')

# Predict on new image
input_img, predicted_img = predict_single_image(
    model, 'path/to/polygon.png', 'red', color_map, transform
)
```

## 📊 Results

The model successfully learns to:
- ✅ Understand polygon shapes from grayscale inputs
- ✅ Interpret text-based color conditions
- ✅ Generate realistic colored outputs
- ✅ Maintain shape boundaries and details
- ✅ Generalize to unseen polygon-color combinations

### Training Curves:
- Training loss decreases steadily from 0.83 to 0.11
- Validation loss decreases from 0.96 to 0.17
- No overfitting observed with early stopping

### Visual Results:
The model produces high-quality colored polygons that match the specified color conditions while preserving the original shape geometry.

## 🔬 Future Improvements

1. **Dataset Expansion**: More polygon types and color variations
2. **Architecture Enhancements**: Attention mechanisms, residual connections
3. **Loss Functions**: Perceptual loss, adversarial training
4. **Advanced Conditioning**: Multiple conditions, texture generation
5. **Model Optimization**: Quantization, pruning for deployment

## 👨‍💻 Author

**Saikiranudayana**
- Ayna ML Intern Assignment
- Implementation Date: August 2025

## 📄 License

This project is created for educational and assignment purposes.