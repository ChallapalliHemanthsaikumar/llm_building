# Feedforward Neural Network Implementation on MNIST Dataset

## Project Overview
This project implements a feedforward neural network to classify handwritten digits using the MNIST dataset. The implementation includes multiple model iterations, from a basic version to an enhanced architecture with improved features.

## Dataset Preparation
- Used MNIST dataset from torchvision
- Dataset contains 60,000 training images and 10,000 test images
- Each image is 28x28 pixels in grayscale
- Images are normalized using transforms.ToTensor()
- Data split: 80% training, 20% testing

## Model Architectures

### 1. Basic MNIST Model
```python
class MNIST(torch.nn.Module):
    - Input layer: 784 neurons (28*28 flattened)
    - Hidden layer: 128 neurons with ReLU activation
    - Output layer: 10 neurons (one for each digit)
```

### 2. Improved MNIST Model
```python
class ImprovedMNIST(torch.nn.Module):
    - Added Batch Normalization
    - Included Dropout (0.3)
    - Same layer structure with better regularization
```

### 3. Enhanced MNIST Model
```python
class EnhancedMNIST(torch.nn.Module):
    - Input layer: 784 neurons
    - First hidden layer: 256 neurons
    - Second hidden layer: 128 neurons
    - Output layer: 10 neurons
    - Features:
        * Batch Normalization after each layer
        * Reduced dropout rate (0.2)
        * ReLU activation
```

## Training Configuration
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001 (reduced from 0.01)
- Batch Size: 32 (increased from 4)
- Epochs: 10-15
- Learning Rate Scheduler: ReduceLROnPlateau

## Model Evaluation
Implemented comprehensive evaluation metrics:
- Confusion Matrix
- Per-digit accuracy
- Precision, Recall, F1-Score
- Overall model accuracy

### Performance Results
1. Basic Model:
   - Overall Accuracy: 92.80%
   - Strong performance on most digits
   - Some confusion between similar digits (5,8)

2. Enhanced Model Features:
   - Learning rate scheduler for better convergence
   - Batch normalization for stable training
   - Dropout for regularization
   - Multiple hidden layers for better feature extraction

## Real-world Testing
Implemented functionality to test with custom handwritten digits:
- Image preprocessing pipeline
- Probability distribution for predictions
- Visualization of results

### Preprocessing Steps:
1. Convert to grayscale
2. Resize to 28x28
3. Normalize pixel values
4. Add batch dimension
5. Transform to match MNIST format

## Model Persistence
- Save model checkpoints including:
  * Model state
  * Optimizer state
  * Training epoch
  * Loss values

## Future Improvements
1. Data Augmentation
2. Deeper architectures
3. Regularization techniques
4. Advanced preprocessing for real-world images
5. Ensemble methods

## Tools and Libraries Used
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn
- PIL (Python Imaging Library)

## Conclusion
The project demonstrates the evolution of a neural network from a basic implementation to an enhanced architecture with improved accuracy and robustness. The final model achieves good performance on both MNIST test data and real-world handwritten digits.
