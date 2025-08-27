# Week 2 Lab: Multiple Variable Linear Regression

This document provides a concise summary of the key concepts and differences between single-variable and multi-variable linear regression, based on the Week 2 Lab notebook for multiple variable linear regression.

## Key Concepts Summary

- **Multiple Variable Linear Regression**: Extends single-variable regression to predict a continuous output using multiple input features (e.g., predicting house price based on size, bedrooms, floors, and age).
- **Model**: Uses a linear combination of features: \( f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \), computed efficiently with vectorized operations (e.g., NumPy's `np.dot`).
- **Cost Function**: Measures average squared error: \( J(\mathbf{w},b) = \frac{1}{2m} \sum_{i=1}^m (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \).
- **Gradient Descent**: Updates all weights (\( \mathbf{w} \)) and bias (\( b \)) to minimize cost using vector operations.
- **Vectorization**: Leverages NumPy for faster computations, avoiding loops for predictions and gradients.
- **Feature Scaling**: Critical for multi-variable regression due to differing feature units (e.g., square feet vs. number of bedrooms).

### Analogies to Remember
- **Single Variable**: Fitting a line through 2D points (one feature, e.g., size vs. price).
- **Multi-Variable**: Fitting a hyperplane in higher dimensions, combining multiple features for better predictions.
- Think of upgrading from predicting house price with just size to using size, bedrooms, floors, and age for a more accurate model.

## Difference Between Single and Multi-Variable Linear Regression

| Aspect                     | Single Variable                     | Multi-Variable                       |
|----------------------------|-------------------------------------|-------------------------------------|
| **Number of Features**     | 1 (e.g., size)                     | Multiple (e.g., size, bedrooms, age) |
| **Model Equation**         | \( h(x) = w_1x + b \)              | \( f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \) |
| **Input Data**             | Scalar per example                 | Vector per example                  |
| **Data Structure**         | Vector \( [x_1, ..., x_m] \)       | Matrix \( \mathbf{X} \) (m × n)     |
| **Weights**                | Single weight \( w_1 \)            | Weight vector \( \mathbf{w} \) (n weights) |
| **Computation**            | Simple multiplication              | Vectorized dot product (np.dot)     |
| **Complexity**             | Simpler, less data needed          | More complex, needs more data       |
| **Feature Scaling**        | Often unnecessary                 | Critical due to different units     |

## Essential Definitions

- **X**: Training matrix (m rows/examples, n columns/features).
- **y**: Target vector (m outputs).
- **w**: Weight vector (n parameters, one per feature).
- **b**: Bias scalar (intercept).
- **Prediction**: \( f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \) (dot product).
- **Cost**: \( J(\mathbf{w},b) = \frac{1}{2m} \sum_{i=1}^m (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \).
- **Gradient**: 
  - For weight \( w_j \): \( \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x^{(i)}_j \).
  - For bias \( b \): \( \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \).
- **m**: Number of examples.
- **n**: Number of features.

## Practice Problems

1. **Given**: Training data \( X = [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]] \), \( y = [460, 232, 178] \).
   - **What’s m?** → 3 (examples).
   - **What’s n?** → 4 (features).
   - **Compute prediction** for first example with \( \mathbf{w} = [0.39, 18.75, -53.36, -26.42] \), \( b = 785.18 \).
     - **Answer**: \( np.dot([2104, 5, 1, 45], \mathbf{w}) + b \approx 460 \) (matches target).
2. **If cost is high and not decreasing**, what’s likely wrong?
   - **Answer**: Learning rate \( \alpha \) too large/small, or features not normalized.
3. **Why use np.dot for predictions?**
   - **Answer**: Faster for large datasets (vectorized computation).

## Real-World Application

- **House Price Prediction**: Tools like Zillow use multiple features (square footage, bedrooms, age, location) in multi-variable linear regression to estimate home values.

## Connection to Advanced Topics

- **Feature Scaling/Normalization**: Critical for multi-variable regression (addressed in later labs).
- **Polynomial Regression**: Extends to non-linear relationships.
- **Neural Networks**: Stacks multiple linear regression units in layers.
- **Regularization**: Prevents overfitting with many features (e.g., Ridge Regression).

## What to Remember

1. **Conceptual Shift**:
   - Single: Fits a line (y = mx + b).
   - Multi: Fits a hyperplane, combining multiple features’ effects.

2. **Vectorization**:
   - Use `np.dot` for predictions (\( \mathbf{w} \cdot \mathbf{x} + b \)) and gradients for efficiency.
   - Example: Prediction = `np.dot(x, w) + b`.

3. **Data Structures**:
   - Single: Scalar inputs, single weight.
   - Multi: Matrix \( \mathbf{X} (m \times n) \), vector \( \mathbf{w} (n,) \), vector \( \mathbf{y} (m,) \).

4. **Cost and Gradient**:
   - Same goal (minimize squared errors), but multi-variable uses vector operations.
   - Compute gradients for each feature’s weight and bias.

5. **Practical Considerations**:
   - **Feature Scaling**: Normalize features to ensure gradient descent converges.
   - **Learning Rate (\( \alpha \))**: Adjust for multi-variable due to more parameters.
   - **Overfitting Risk**: More features increase risk; consider regularization.

6. **Implementation**:
   - Single: Simple scalar math or loops.
   - Multi: Use matrix operations (e.g., \( \mathbf{X} @ \mathbf{w} + b \)).
   - Ensure correct shapes: \( \mathbf{X} (m,n) \), \( \mathbf{w} (n,) \), \( \mathbf{y} (m,) \).

7. **Real-World Intuition**:
   - Single: Quick models for one dominant factor.
   - Multi: Better for complex problems with multiple influencing factors.

8. **Analogy**:
   - Single: Predicting height with just weight (limited).
   - Multi: Predicting height with weight, age, gender, diet (more accurate).

## Quick Reference

### Formula Sheet
| Concept            | Formula                                                                 | Key Variables                                  |
|--------------------|-------------------------------------------------------------------------|------------------------------------------------|
| Model Prediction   | \( f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \)                   | \( \mathbf{w} \): weight vector, \( \mathbf{x} \): feature vector |
| Cost Function      | \( J = \frac{1}{2m} \sum (f(\mathbf{x}^{(i)}) - y^{(i)})^2 \)           | \( m \): examples, \( i \): example index      |
| Gradient \( w_j \) | \( \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum (f(\mathbf{x}^{(i)}) - y^{(i)}) x^{(i)}_j \) | \( j \): feature index |
| Gradient \( b \)   | \( \frac{\partial J}{\partial b} = \frac{1}{m} \sum (f(\mathbf{x}^{(i)}) - y^{(i)}) \) |                |
| Gradient Descent   | \( \mathbf{w} = \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}} \), \( b = b - \alpha \cdot \frac{\partial J}{\partial b} \) | \( \alpha \): learning rate |
| Vectorized Predict | `np.dot(x, w) + b`                                                     | NumPy for speed                                |

### Problem-Solving Checklist
- Multiple features? → Use matrices/vectors.
- Check shapes: \( \mathbf{X} (m,n) \), \( \mathbf{w} (n,) \), \( \mathbf{y} (m,) \).
- Cost decreasing? → Tune \( \alpha \), run more iterations.
- Overfitting? → Too many features; consider regularization.
- Underfitting? → Add features or complexity.

### Common Mistakes to Avoid
- ❌ Looping over features (use `np.dot` instead).
- ❌ Mismatched shapes for \( \mathbf{X} \), \( \mathbf{w} \), or \( \mathbf{y} \).
- ❌ Incorrect learning rate \( \alpha \) (too high = diverges, too low = slow).
- ❌ Poor initialization of \( \mathbf{w} \), \( b \) (start near zero or optimal).
- ❌ Ignoring feature units (e.g., sqft vs. bedrooms requires scaling).

## 10-Second Revision
"Single: one feature, simple line, scalar math. Multi: many features, hyperplane, vectorized dot products, needs scaling."

---

This README is derived from the Week 2 Lab notebook and the provided notes, tailored for clarity and quick reference on GitHub.

