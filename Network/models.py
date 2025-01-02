import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset


# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, image_features, tabular_features, labels):
        self.image_features = image_features
        self.tabular_features = tabular_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image_features[idx], self.tabular_features[idx], self.labels[idx]


# DenseNet Backbone
class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(DenseNetFeatureExtractor, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features  # Extract DenseNet feature layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to a fixed size

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x


# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5))
        output = torch.bmm(attention_scores, V)
        return output


# Feature Pyramid Network (FPN)
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.fpn_layers = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, features):
        fpn_outputs = []
        for idx, feature in enumerate(features):
            fpn_output = self.fpn_layers[idx](feature)
            if idx > 0:
                fpn_output += self.upsample(fpn_outputs[-1])
            fpn_outputs.append(fpn_output)
        return fpn_outputs

class MedFusionNet(nn.Module):
    def __init__(self, tabular_input_dim, output_dim):
        super(MedFusionNet, self).__init__()
        # DenseNet for image features
        self.densenet = DenseNetFeatureExtractor()

        # FPN for multi-scale image features
        self.fpn = FeaturePyramidNetwork(in_channels_list=[512, 256, 128], out_channels=128)

        # Self-Attention for tabular data
        self.self_attention = SelfAttention(input_dim=tabular_input_dim, output_dim=64)

        # Fusion Layer
        self.fc_fusion = nn.Linear(128 + 64, 128)

        # Final Layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, image_input, tabular_input):
        # Image features
        image_features = self.densenet(image_input)
        fpn_features = self.fpn([image_features])[-1]  # Use the last output of FPN

        # Tabular features with attention
        tabular_features = self.self_attention(tabular_input).mean(dim=1)

        # Concatenate features
        fused_features = torch.cat((fpn_features, tabular_features), dim=1)
        fusion_output = torch.relu(self.fc_fusion(fused_features))

        # Final classification layers
        x = torch.relu(self.fc1(fusion_output))
        output = torch.sigmoid(self.fc2(x))  # Sigmoid for binary/multilabel classification
        return output


# DataLoader
def create_hybrid_dataloader(image_features, tabular_features, labels, batch_size=32):
    dataset = CustomImageDataset(image_features, tabular_features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)