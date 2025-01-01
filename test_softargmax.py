import torch
import torch.nn.functional as F

# 定义 Softmin 函数，确保数值稳定
def softmin(corr_map, tau=0.001):
    batch, h, w = corr_map.shape
    corr_map_flat = corr_map.view(batch, -1)
    corr_map_flat = corr_map_flat - corr_map_flat.max(dim=-1, keepdim=True).values
    softmin_probs_flat = F.softmax(-corr_map_flat / tau, dim=-1)
    return softmin_probs_flat.view(batch, h, w)

# 示例相关图
batch, h, w = 4, 10, 10
corr_map = torch.rand(batch, h, w, requires_grad=True)

# 计算 Softmin
tau = 0.001
softmin_probs = softmin(corr_map, tau)

# 定义简单的损失函数（例如，平滑坐标差异）
y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
x_coords = x_coords.to(corr_map.device).float()
y_coords = y_coords.to(corr_map.device).float()

x_mean = torch.sum(softmin_probs * x_coords[None, :, :], dim=(-2, -1))
y_mean = torch.sum(softmin_probs * y_coords[None, :, :], dim=(-2, -1))

# 平滑坐标的欧几里得距离损失
target_x, target_y = torch.tensor([5.0, 5.0]).to(corr_map.device)
loss = torch.mean(torch.abs((x_mean - target_x)) + torch.abs((y_mean - target_y)))

# 反向传播
loss.backward()

# 验证梯度
print("Loss:", loss.item())
print("Gradient on corr_map:", corr_map.grad)
