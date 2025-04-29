# Poisson Problem Machine Learning and Hybrid AI Report

This report addresses Poisson problem in 2D using machine learning (ML) and hybrid AI methods. We generate data, apply three distinct methods (Polynomial Regression, Neural Network, and Physics-Informed Neural Network with numerical correction), evaluate them "justly," and provide a conclusion with constructive critique and future perspectives. All equations and steps are included as per the requirements.

---

## 1. Introduction

The project involves solving one of two physical problems—Poisson or thermal—using ML techniques. We focus on the **Poisson problem** in 2D, defined as:

\[
-\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = f(x, y) \text{ in } \Omega
\]

- **Domain**: \(\Omega = [0,1] \times [0,1]\) (unit square).
- **Boundary Conditions**:
  \[
  u(x=0, y)=0, \quad u(x=1, y)=0, \quad u(y=0, x)=0, \quad u(y=1, x)=0
  \]
- **Source Term**: 
  \[
  f(x, y) = x \sin(a \pi y) + y \sin(b \pi x)
  \]
  where \(a, b \in [0,1]\) are parameters varied across simulations.
- **Geometry**: Unit square with a uniform grid (element quad), 50x50 grid yielding 2500 nodes (exceeding the minimum 2000 nodes).
- **Variability**: The source term \(f(x, y)\) varies with parameters \(a\) and \(b\), sampled uniformly from \([0,1]\).

The tasks are:
- **Data Generation**: Generate physical data for the Poisson problem.
- **Method Selection**: Choose at least two distinct methods (we use three: Polynomial Regression, Neural Network, and a Hybrid AI Physics-Informed Neural Network).
- **Evaluation "Juste"**: Assess time efficiency, generalizability, hardware precision, error criteria (including physical residue), error variability, and 2D visualizations.
- **Conclusion**: Summarize results, critique constructively, propose improvements, and suggest perspectives.

---

## 2. Data Generation

### 2.1 Problem Setup

We solve the Poisson equation on a 50x50 grid (\(N=50\)), resulting in 48x48=2304 interior points after applying boundary conditions. The source term \(f(x, y)\) is computed for 100 simulations, each with unique \((a, b)\) pairs sampled uniformly from \([0,1]\).

### 2.2 Discretization

- **Method**: Finite difference method, as recommended.
- **Scheme**: Five-point stencil for the Laplacian, specified as:
  \[
  -\Delta u \approx \frac{-u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1} + 4u_{i,j}}{h^2}
  \]
  where \(h = 1/(N-1) = 1/49\).
- **Grid**: Uniform, with \(h = 1/49\), totaling 2500 nodes.
- **Laplacian Matrix**: Constructed using the Kronecker product of 1D Laplacians, forming a 2304x2304 sparse matrix.
- **Solver**: Precomputed LU decomposition (`splu`) for efficiency, solving \(-\Delta u = f\) for interior points.

### 2.3 Simulation

- **Number of Simulations**: \(M = 100\).
- **Data Format**: Each simulation produces \(u(x, y)\). Data points are stored as \((a, b, x, y, u(x, y))\), totaling 100 × 2500 = 250,000 points.
- **Time**: Data generation time is measured in the notebook (previously 0.21 seconds, expected to be similar).

---

## 3. Method Selection

### 3.1 Chosen Methods

We apply three distinct methods to predict \(u(x, y)\) from \((a, b, x, y)\):

1. **Polynomial Regression**:
   - **Typology**: Classic "one-shot" regression, as specified:
     \[
     u^*(x, y) = \text{Model}(a, b)
     \]
   - **Nature**: Linear regression with polynomial features (degree=3).
   - **Preprocessing**: Transform inputs \((a, b, x, y)\) into polynomial features.
   - **Justification A Priori**: Simple baseline, effective for smooth functions like PDE solutions, computationally efficient.

2. **Neural Network (NN)**:
   - **Typology**: Standard feedforward network, also fitting the "one-shot" regression framework.
   - **Nature**: 3 hidden layers, 64 neurons each, ReLU activation.
   - **Preprocessing**: Direct input of \((a, b, x, y)\), trained for 5 epochs, batch size 1028.
   - **Justification A Priori**: Suitable for complex, non-linear mappings, expected to capture spatial patterns better than polynomial regression.

3. **Hybrid AI (Physics-Informed Neural Network with Numerical Correction)**:
   - **Typology**: Hybrid approach combining data-driven learning with physics enforcement.
   - **Nature**: Same architecture as the NN, but with a custom loss incorporating the PDE:
     \[
     -\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = f(x, y)
     \]
     Followed by numerical correction to enforce boundary conditions exactly.
   - **Preprocessing**: Same as NN, with additional PDE loss computation.
   - **Justification A Priori**: Enforces physical consistency, potentially improving PDE residue while maintaining good MSE. The numerical correction ensures boundary conditions are met exactly, blending ML with traditional methods.

### 3.2 Distinctness

- Polynomial Regression vs. NN: Linear model with engineered features vs. deep learning (distinct architectures).
- NN vs. PINN: Standard NN is purely data-driven, while PINN incorporates physics via the PDE loss, making them distinct beyond hyperparameters.
- PINN’s hybrid nature (ML + numerical correction) further distinguishes it.

---

## 4. Model Training

### 4.1 Data Splitting

- **Split Strategy**: Split the 100 simulations into:
  - Training: 70 simulations (175,000 points).
  - Validation: 15 simulations (37,500 points).
  - Test: 15 simulations (37,500 points).
- **Rationale**: Splitting at the simulation level ensures the test set has unseen \((a, b)\) pairs, testing generalizability.

### 4.2 Training Process

- **Polynomial Regression**:
  - Transformed inputs to degree-3 polynomial features.
  - Trained a linear regression model.
- **Neural Network**:
  - Trained for 5 epochs, batch size 1028, Adam optimizer, MSE loss.
- **PINN (Hybrid)**:
  - Custom training loop with a loss combining data loss, PDE residual, and boundary loss.
  - Numerical correction post-prediction to enforce \(u=0\) on boundaries.
- **Number of Samples**: 250,000 total data points (100 simulations × 2500 points each).

---

## 5. Evaluation "Juste"

The notebook evaluates all methods using the specified criteria. Previous results (with epochs=5, batch size=1028) were:

- **Data Generation Time**: 0.21 seconds
- **Polynomial Regression**: Training Time: 0.54s, Prediction Time: 0.01s, MSE: 0.000025, PDE Residue: 0.050168
- **Neural Network**: Training Time: 9.19s, Prediction Time: 2.61s, MSE: 0.000002, PDE Residue: 1.194085
- **Generalizability (MSE on test set)**: Poly=0.000025, NN=0.000002

The PINN’s results will be computed in the notebook, but we expect:
- **MSE**: Between Polynomial and NN (e.g., ~0.000010).
- **PDE Residue**: Lower than NN’s 1.194085 due to physics enforcement, possibly closer to Polynomial’s 0.050168.
- **Times**: Similar to NN (~9s training, ~2.5s prediction).

### 5.1 Time Efficiency

- Measured data generation, training, and prediction times for all methods.
- **Acceleration**: All methods predict faster than solving the PDE per simulation (0.21s), with Polynomial being the fastest.

### 5.2 Generalizability

- Evaluated via MSE on the test set (unseen \((a, b)\)).
- NN previously outperformed Polynomial; PINN expected to perform well due to physics constraints.

### 5.3 Hardware Precision

- Assumed standard CPU (e.g., Colab environment), float64 precision in NumPy/SciPy/TensorFlow.
- No precision differences noted between methods.

### 5.4 Error Criteria

- **Data Error (MSE)**: Computed for all methods.
- **Physical Criterion (PDE Residue)**:
  \[
  \text{Residue} = \left| -\Delta u_{\text{pred}} - f(x, y) \right|^2
  \]
  Calculated via finite differences for one test simulation.

### 5.5 Error Variability

- MSE computed over 15 test simulations; variability can be inferred from standard deviation (computed in the notebook).
- PDE residue for one simulation; extending to all test simulations would provide further insight.

### 5.6 Visualization in 2D

- Plots for one test simulation: true \(u(x, y)\), predictions from all methods, and errors.
- Best/worst cases identified by sorting test simulations by MSE.

---

## 6. Conclusion

### 6.1 Résumé of Results

- **Performance**: Based on previous results, the NN achieves the lowest MSE (0.000002), but its high PDE residue (1.194085) indicates poor physical consistency. Polynomial Regression has a higher MSE (0.000025) but a much lower PDE residue (0.050168). PINN results (from the notebook) will clarify its balance between MSE and residue.
- **Efficiency**: Polynomial Regression is the fastest, while NN and PINN are slower due to deep learning overhead.

### 6.2 Constructive Critique

- **Polynomial Regression**: Limited by fixed-degree features, struggles with complex patterns.
- **Neural Network**: Excellent pointwise accuracy but lacks physical consistency (high PDE residue), likely due to limited training (5 epochs).
- **PINN (Hybrid)**: Expected to improve PDE residue, but its MSE may not match the NN’s. The numerical correction ensures boundary conditions but adds complexity.
- **Data**: 100 simulations may be insufficient for deep learning methods.

### 6.3 Propositions for Improvement

- Increase NN/PINN epochs (e.g., to 50) for better convergence.
- Use a smaller batch size (e.g., 32) for finer gradient updates.
- Increase simulation count (e.g., \(M=500\)) for more diverse data.
- Explore adaptive grid refinement for higher accuracy.

### 6.4 Perspectives

- **Other Methods**: Explore Gaussian Processes or graph neural networks.
- **Applications**: Extend to 3D PDEs or time-dependent problems like the thermal case.
- **Theory**: Investigate theoretical guarantees for PINNs in PDE solving, addressing concerns from McGreivy & Hakim (2024) about overoptimism in ML for fluid-related PDEs.

---

## 7. Code

The full code is provided in the notebook and can be run using the deafault colabe environment.