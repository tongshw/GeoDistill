from torch import nn


class RotationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_angle, gt_angle):
        # pred_angle: 模型预测角度
        # gt_angle: 真实角度
        return self.mse_loss(pred_angle, gt_angle)