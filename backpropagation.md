# Understanding Backpropagation with Visualizations

## Introduction
This document explains the implementation of backpropagation on a simple XOR dataset, with visualizations to understand how neural networks learn.

## Dataset
We use the XOR dataset as it's small and demonstrates non-linear learning:
```python
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
```

## Network Architecture
- Input Layer: 2 neurons
- Hidden Layer: 2 neurons with sigmoid activation
- Output Layer: 1 neuron with sigmoid activation

## Implementation Steps

### 1. Network Initialization
```python
input_size = 2
hidden_size = 2
output_size = 1
```

### 2. Forward Pass
- Hidden Layer: z1 = X·W1 + b1
- Hidden Activation: a1 = sigmoid(z1)
- Output Layer: z2 = a1·W2 + b2
- Output Activation: a2 = sigmoid(z2)

### 3. Loss Computation
Mean Squared Error (MSE) Loss:
```python
loss = np.mean((y - a2) ** 2)
```

### 4. Backward Pass
Compute gradients:
- Output Layer Error: d_a2 = (a2 - y) * sigmoid_derivative(a2)
- Hidden Layer Error: d_a1 = (d_a2·W2ᵀ) * sigmoid_derivative(a1)

### 5. Weight Updates
Apply gradients with learning rate:
```python
W2 -= learning_rate * d_W2
b2 -= learning_rate * d_b2
W1 -= learning_rate * d_W1
b1 -= learning_rate * d_b1
```

## Visualizations

### 1. Loss Curve
```python
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
```
This shows how well the network is learning over time.

### 2. Hidden Layer Activations
```python
for neuron in range(hidden_size):
    plt.plot(epochs, activations[:, neuron])
plt.title('Hidden Layer Neuron Activations')
```
Visualize how hidden neurons respond to inputs.

### 3. Weight Changes
```python
plt.plot(epochs, W1_history)
plt.title('Weight Evolution During Training')
```
Track how weights adapt during training.

### 4. Decision Boundary
```python
def plot_decision_boundary():
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
```
Visualize how the network separates the input space.

## Key Visualization Tips

1. **Loss Tracking**
   - Plot loss every N epochs
   - Use log scale for better visibility
   - Add moving average for smoother curves

2. **Weight Distribution**
   - Histogram of weights
   - Track mean and variance
   - Watch for vanishing/exploding gradients

3. **Activation Patterns**
   - Plot activation distributions
   - Check for dead neurons (always 0)
   - Monitor saturation

4. **Gradient Flow**
   - Visualize gradient magnitudes
   - Check for healthy backpropagation
   - Identify gradient vanishing/exploding

## Best Practices

1. **Initialization**
   - Use small random weights
   - Initialize biases to zero
   - Consider Xavier/He initialization

2. **Learning Rate**
   - Start with small learning rate (0.1)
   - Monitor loss for stability
   - Adjust based on loss curve

3. **Monitoring**
   - Track multiple metrics
   - Save visualizations periodically
   - Compare different runs

## Debugging Tips

1. **Loss Not Decreasing**
   - Check gradient calculations
   - Verify weight updates
   - Adjust learning rate

2. **Unstable Training**
   - Reduce learning rate
   - Check for NaN values
   - Monitor gradient magnitudes

3. **Poor Convergence**
   - Increase hidden layer size
   - Adjust initialization
   - Try different activation functions

## Conclusion
Visualizing the training process helps understand:
- How the network learns
- Where problems occur
- When training is complete
- How to tune hyperparameters
