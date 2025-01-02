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

test_loader = create_hybrid_dataloader(X_image_test, X_tabular_test, y_test)

# Initialize the MedFusionNet model
model = MedFusionNet(tabular_input_dim=X_tabular_train.shape[1], output_dim=1)

# Evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for img_batch, tab_batch, lbl_batch in test_loader:
        predictions = model(img_batch, tab_batch).round()
        all_predictions.append(predictions)
        all_labels.append(lbl_batch)

    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate Metrics
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    gmean = geometric_mean_score(all_labels, all_predictions)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("G-Mean:", gmean)

    # Confusion Matrix Plot
    cm_fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True", color="Count"),
                       title="Confusion Matrix")
    cm_fig.show()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
    roc_fig.update_layout(title="Receiver Operating Characteristic (ROC) Curve",
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          showlegend=True)
    roc_fig.show()

    # Precision-Recall Curve Plot
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_predictions)

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name="Precision-Recall Curve"))
    pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
                         showlegend=True)
    pr_fig.show()

    # Save plots as HTML
    cm_fig.write_html("/kaggle/working/confusion_matrix.html")
    roc_fig.write_html("/kaggle/working/roc_curve.html")
    pr_fig.write_html("/kaggle/working/precision_recall_curve.html")

# Optional: Handling imbalance using SMOTE for tabular data
smote = SMOTE(random_state=42)
tabular_resampled, labels_resampled = smote.fit_resample(X_tabular_train, y_train)

# After applying SMOTE, re-train the model on the resampled tabular data
tabular_train_loader = create_hybrid_dataloader(
    X_image_train, torch.tensor(tabular_resampled), torch.tensor(labels_resampled)
)

# Retrain the model similarly
