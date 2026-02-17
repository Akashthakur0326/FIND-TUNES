import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# SPECTROGRAM MODEL (Shallow CNN)
# ==========================================
class AudioSiameseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Conv Block 1 (learns Local time–frequency edges)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Conv Block 2 (learns Harmonic stacks)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Conv Block 3 (no pooling to preserve melody resolution)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, embed_dim),
        )

    def forward_one(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, anchor, positive=None, negative=None, *args):
        # Enables both ONNX tracing (single input) and Triplet Training (3 inputs)
        if positive is None or negative is None:
            return self.forward_one(anchor)
        return self.forward_one(anchor), self.forward_one(positive), self.forward_one(negative)


# ==========================================
# PITCH MODEL (CRNN)
# ==========================================
class CRNN(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, embed_dim)
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # (Batch, Seq, Features) for LSTM
        
        # Note: self.lstm.flatten_parameters() is removed for safe ONNX export
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        return F.normalize(self.fc(out), p=2, dim=1)
        
    def forward(self, anchor, positive=None, negative=None, *args):
        if positive is None or negative is None:
            return self.forward_one(anchor)
        return self.forward_one(anchor), self.forward_one(positive), self.forward_one(negative)