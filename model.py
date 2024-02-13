import torch.nn as nn


class CNN3DModel(nn.Module):
    def __init__(self, in_channels, n_classes, hidden_dim):
        print("Creating model (1/1)")
        super(CNN3DModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv3d(in_channels, hidden_dim // 4, kernel_size=3, padding='same')
        self.conv2 = nn.Conv3d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding='same')
        self.conv3 = nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=3, padding='same')

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc2 = nn.Linear(hidden_dim, n_classes)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))

        x = self.gap(x)
        x = x.view(-1, self.hidden_dim)
        x = self.fc2(x)

        return x
