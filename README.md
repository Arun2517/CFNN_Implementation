# CFNN for Breast Cancer Classification Using Excel

A Closed-Form Neural Network (CFNN) implementation for the **Wisconsin Breast Cancer (WBC) Dataset** using **Microsoft Excel**, **Python**, and **xlwings**. The model employs a **single hidden layer** and computes the output weights analytically using the **pseudo-inverse**, eliminating the need for iterative backpropagation.

The implementation demonstrates that high classification accuracy can be achieved with a simple neural network architecture and closed-form learning directly within Microsoft Excel.

---

## Overview

This project implements a **Single Hidden Layer Closed-Form Neural Network (CFNN)** with:

- Single hidden layer
- 20 hidden neurons
- Pseudo-inverse-based output weight computation
- Excel-based implementation using xlwings
- Binary classification on the Wisconsin Breast Cancer Dataset

---

## Features

- 📊 Runs directly from Microsoft Excel
- 🐍 Powered by Python and xlwings
- ⚡ Closed-form learning (No Backpropagation)
- 🧮 pseudo-inverse solution
- 🧠 Single hidden layer architecture
- 📈 Binary classification
- 🚀 Fast training
- 📋 Excel User-Defined Functions (UDFs)

---

## Dataset

**Wisconsin Breast Cancer (WBC) Dataset**

The dataset is used to classify breast tumors into:

- Benign
- Malignant

---

## Network Architecture

Input Layer

↓

Hidden Layer (20 Neurons)

↓

Activation Function

↓

Pseudo-Inverse Learning

↓

Output Layer

---

## Learning Method

The hidden-to-output weights are computed analytically using:

\[
W = H^{+}Y
\]

where

- **H** = Hidden layer output matrix
- **H⁺** = Moore–Penrose pseudo-inverse of H
- **Y** = Target matrix

No gradient descent or backpropagation is required.

---

## Results

| Metric | Accuracy |
|---------|---------:|
| Training Accuracy | **97.37%** |
| Testing Accuracy | **96.49%** |

---

## Technologies Used

- Microsoft Excel
- Python 3.x
- xlwings
- NumPy
- pandas

---

## Requirements

- Microsoft Excel
- Python 3.10 or later
- xlwings
- NumPy
- pandas

Install the required packages:

```bash
pip install xlwings numpy pandas
```

---

## Getting Started

1. Clone the repository.

```bash
git clone https://github.com/Arun2517/<repository-name>.git
```

2. Install the required Python packages.

3. Open the Excel workbook.

4. Install **xlwings Lite** from **Excel → Add-ins**.

5. Paste the required Python functions into the xlwings editor.

6. Save the workbook.

7. Use the Excel functions to perform training and prediction.

---

## Applications

- Breast Cancer Classification
- Medical Data Analysis
- Educational Demonstrations
- Rapid Machine Learning Prototyping
- Closed-Form Neural Network Research

---

## Advantages

- No Backpropagation
- No Learning Rate
- No Epochs
- Deterministic Learning
- Fast Training
- Simple Architecture
- Easy to Use from Microsoft Excel

---

## Future Work

- Multi-class classification
- Additional activation functions
- Ensemble CFNN
- Larger medical datasets
- Deep Closed-Form Networks
- Model visualization within Excel

---

## Citation

If you use this repository in your research, please cite:

```text
Arun G.
Closed-Form Neural Network for Breast Cancer Classification Using Microsoft Excel.
GitHub Repository.
```

---

## Author

**Arun G**
