import torch

for epoch in range(3):
    torch.manual_seed(42)  # 每次都用相同的种子初始化
    rand_num = torch.rand(1)
    print(f"Epoch {epoch}, Random Number: {rand_num}")
