import torch


def generate_mask(feature_map, r):
    """
    生成一个只包含0和1的mask,值为1的元素个数/所有元素格式=r

    参数:
    feature_map (torch.Tensor): 输入的feature map,shape为(batch, channel, height, width)
    r (float): 需要保留的比例,范围为(0, 1)

    返回:
    mask (torch.Tensor): 生成的mask,shape为(batch, 1, height, width)
    """
    batch_size, channel, height, width = feature_map.shape

    # 计算每个位置的L2范数
    l2_norms = torch.norm(feature_map, p=2, dim=1, keepdim=True)
    k = int(r * height * width)
    # 对L2范数进行排序,获取前r个位置的索引
    _, indices = torch.topk(l2_norms.view(batch_size, -1), k, dim=1, largest=True)

    # 生成mask
    mask = torch.ones_like(l2_norms)
    mask.view(batch_size, -1).scatter_(1, indices, 0)

    return mask

#
# # 假设feature_map的shape为(4, 16, 32, 32)
# feature_map = torch.randn(4, 16, 32, 32)
# r = 0.2
#
# mask = generate_mask(feature_map, r)
# print(mask.shape)  # torch.Size([4, 1, 32, 32])
# print(mask.sum() / mask.numel())  # 0.2
# print(mask[0])