import torch
import torch.nn as nn
from torchvision import models

class MultiModalFusionModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=2):
        super(MultiModalFusionModel, self).__init__()
        
        # 1. 기상 시계열 LSTM Branch
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # 2. 위성 이미지 CNN Branch (Pretrained ResNet18 백본)
        # weights=None (또는 pretrained=False)으로 초기화하여 로드 준비
        resnet = models.resnet18(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # CNN 임베딩 벡터 차원 축소 (512 -> 64)
        self.cnn_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Fusion Output Layer
        # LSTM 특징(64) + CNN 특징(64) = Concat 특징(128)
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x_numeric, x_image, sunshine_mask=None):
        # A. LSTM Branch
        out_num, _ = self.lstm1(x_numeric)
        out_num = self.dropout1(out_num)
        out_num, _ = self.lstm2(out_num)
        feat_numeric = self.dropout2(out_num[:, -1, :]) # (batch_size, 64)
        
        # B. CNN Branch
        feat_img = self.cnn_backbone(x_image) # (batch_size, 512, 1, 1)
        feat_img = torch.flatten(feat_img, 1) # (batch_size, 512)
        feat_img = self.cnn_fc(feat_img)       # (batch_size, 64)
        
        # C. Fusion
        fused = torch.cat([feat_numeric, feat_img], dim=1) # (batch_size, 128)
        pred = self.fusion_fc(fused) # (batch_size, 2)
        
        # D. 물리 마스킹
        if sunshine_mask is not None:
            pred[:, 0] = pred[:, 0] * sunshine_mask
            
        return pred

