import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# ==========================================================
# Load Dataset
# ==========================================================

data = pd.read_csv("/Users/arun/Desktop/Soman Sir Notes/latest_neural_breast cancer/wdbc.data.csv")

# ==========================================================
# Features (First 10 Features)
# ==========================================================

X = data.iloc[:, 0:10].values

# ==========================================================
# Class Labels
# ==========================================================

Y = data["class"].values

# ==========================================================
# Normalize Features
# ==========================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================================
# One Hot Encoding
# ==========================================================

encoder = OneHotEncoder(sparse_output=False)

YOH = encoder.fit_transform(Y.reshape(-1, 1))

# ==========================================================
# Train Test Split (80-20)
# ==========================================================

XTrain, XTest, YTrain, YTest, YTrainOH, YTestOH = train_test_split(
    X,
    Y,
    YOH,
    test_size=0.2,
    random_state=1,
    stratify=Y
)

print("\n=========================================")
print(" SINGLE HIDDEN LAYER PSEUDOINVERSE")
print("=========================================")

neurons = [10, 20, 50, 100, 200]

for hidden in neurons:

    np.random.seed(1)

    # Random Matrices

    Q1 = np.random.randn(XTrain.shape[1], hidden)
    U1 = np.random.randn(YTrainOH.shape[1], hidden)

    # Hidden Target

    Z1_tilde = np.tanh(XTrain @ Q1) + np.tanh(YTrainOH @ U1)

    # Hidden Weight

    W1 = np.linalg.pinv(XTrain) @ Z1_tilde

    # Hidden Output

    Z1 = np.tanh(XTrain @ W1)

    # Output Weight

    W0 = np.linalg.pinv(Z1) @ YTrainOH

    # Training Prediction

    YPredTrain = Z1 @ W0

    trainClass = np.argmax(YPredTrain, axis=1)

    trainAcc = accuracy_score(YTrain, trainClass) * 100

    # Testing

    Z1Test = np.tanh(XTest @ W1)

    YPredTest = Z1Test @ W0

    testClass = np.argmax(YPredTest, axis=1)

    testAcc = accuracy_score(YTest, testClass) * 100

    print(f"Neurons = {hidden:3d} | Train = {trainAcc:6.2f}% | Test = {testAcc:6.2f}%")

print("=========================================")

# ==========================================================
# TWO HIDDEN LAYERS
# ==========================================================

print("\n=========================================")
print(" TWO HIDDEN LAYER PSEUDOINVERSE")
print("=========================================")

architectures = [
    [20,20],
    [20,50],
    [50,50],
    [50,100],
    [100,100]
]

for arch in architectures:

    h1 = arch[0]
    h2 = arch[1]

    np.random.seed(1)

    # Layer 1

    Q1 = np.random.randn(XTrain.shape[1], h1)
    U1 = np.random.randn(YTrainOH.shape[1], h1)

    Z1_tilde = np.tanh(XTrain @ Q1) + np.tanh(YTrainOH @ U1)

    W1 = np.linalg.pinv(XTrain) @ Z1_tilde

    Z1 = np.tanh(XTrain @ W1)

    # Layer 2

    Q2 = np.random.randn(h1, h2)
    U2 = np.random.randn(YTrainOH.shape[1], h2)

    Z2_tilde = np.tanh(Z1 @ Q2) + np.tanh(YTrainOH @ U2)

    W2 = np.linalg.pinv(Z1) @ Z2_tilde

    Z2 = np.tanh(Z1 @ W2)

    # Output Layer

    W0 = np.linalg.pinv(Z2) @ YTrainOH

    # Training

    YPredTrain = Z2 @ W0

    trainClass = np.argmax(YPredTrain, axis=1)

    trainAcc = accuracy_score(YTrain, trainClass) * 100

    # Testing

    Z1Test = np.tanh(XTest @ W1)

    Z2Test = np.tanh(Z1Test @ W2)

    YPredTest = Z2Test @ W0

    testClass = np.argmax(YPredTest, axis=1)

    testAcc = accuracy_score(YTest, testClass) * 100

    print(f"Architecture = [{h1:3d} {h2:3d}] | Train = {trainAcc:6.2f}% | Test = {testAcc:6.2f}%")

print("=========================================")

# ==========================================================
# Best Architecture Example
# ==========================================================

h1 = 50
h2 = 50

np.random.seed(1)

Q1 = np.random.randn(XTrain.shape[1], h1)
U1 = np.random.randn(YTrainOH.shape[1], h1)

Z1_tilde = np.tanh(XTrain @ Q1) + np.tanh(YTrainOH @ U1)

W1 = np.linalg.pinv(XTrain) @ Z1_tilde

Z1 = np.tanh(XTrain @ W1)

Q2 = np.random.randn(h1, h2)
U2 = np.random.randn(YTrainOH.shape[1], h2)

Z2_tilde = np.tanh(Z1 @ Q2) + np.tanh(YTrainOH @ U2)

W2 = np.linalg.pinv(Z1) @ Z2_tilde

Z2 = np.tanh(Z1 @ W2)

W0 = np.linalg.pinv(Z2) @ YTrainOH

# ==========================================================
# Final Testing
# ==========================================================

Z1Test = np.tanh(XTest @ W1)

Z2Test = np.tanh(Z1Test @ W2)

YPred = Z2Test @ W0

predClass = np.argmax(YPred, axis=1)

accuracy = accuracy_score(YTest, predClass) * 100

print(f"\nFinal Test Accuracy = {accuracy:.2f}%")

# ==========================================================
# Matrix Sizes
# ==========================================================

print("\nMatrix Sizes")
print("XTrain   :", XTrain.shape)
print("Q1       :", Q1.shape)
print("U1       :", U1.shape)
print("Z1_tilde :", Z1_tilde.shape)
print("W1       :", W1.shape)
print("Z1       :", Z1.shape)
print("W0       :", W0.shape)
