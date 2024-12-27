import torch

# 构造输入张量，要求梯度
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

# 第一项：不可导的 argmax 操作
argmax_indices = torch.argmax(input_tensor, dim=1)  # 返回最大值索引

# 第二项：可导的常规损失
regular_loss = input_tensor.sum()  # 例如普通的和操作

# 总的 loss
try:
    total_loss = regular_loss + argmax_indices.sum()  # 合并两部分
    total_loss.backward()
    print("梯度:", input_tensor.grad)
except RuntimeError as e:
    print("错误信息:", e)
