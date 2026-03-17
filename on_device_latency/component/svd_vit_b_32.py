import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}

class SVD_ViTMLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0, act='gelu', ratio=0.25):
        super().__init__()
        # Tính toán rank thấp
        low_rank = int(in_features * hidden_features * ratio / (in_features + hidden_features))

        # Thay thế Linear bằng hai tầng low-rank
        self.fc1_v = nn.Linear(in_features, low_rank, bias=False)
        self.fc1_u = nn.Linear(low_rank, hidden_features, bias=True)  # giữ bias như original

        self.act = nn.GELU(approximate='none')

        self.fc2_v = nn.Linear(hidden_features, low_rank, bias=False)
        self.fc2_u = nn.Linear(low_rank, in_features, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1_u(self.fc1_v(x))
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2_u(self.fc2_v(x))
        x = self.dropout(x)
        return x
