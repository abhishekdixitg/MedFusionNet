import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader, TensorDataset
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split

from Network.models import MedFusionNet

# File path and label columns
file_path = '/kaggle/input/risk-factors-cervical-cancer/risk_factors_cervical_cancer.csv'
label_column = "Dx:Cancer"

# Preprocessing for image and tabular data
def preprocess_data(file_path, label_column):
    # Replace this with your actual data loading and preprocessing steps
    image_data = torch.randn(100, 3, 224, 224)  # Example: Replace with actual image data loading
    tabular_data = torch.randn(100, 10)         # Example: Replace with actual tabular data
    labels = torch.randint(0, 2, (100, 1))      # Binary labels for 100 samples
    return image_data, tabular_data, labels

# Load and preprocess data
image_features, tabular_features, labels = preprocess_data(file_path, label_column)

# Split the dataset into training and testing
X_image_train, X_image_test, X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
    image_features, tabular_features, labels, test_size=0.2, random_state=42
)

# Create DataLoader
def create_hybrid_dataloader(image_data, tabular_data, labels, batch_size=32):
    dataset = TensorDataset(image_data, tabular_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_hybrid_dataloader(X_image_train, X_tabular_train, y_train)


# Initialize the MedFusionNet model
model = MedFusionNet(tabular_input_dim=X_tabular_train.shape[1], output_dim=1)

# Define optimizer and loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for img_batch, tab_batch, lbl_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(img_batch, tab_batch)
        loss = criterion(predictions, lbl_batch.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")