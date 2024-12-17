import numpy as np


def visualize_grid_indexing(img_size=512, grid_size=16):
    grid_width = img_size // grid_size

    # 创建索引矩阵
    index_matrix = np.zeros((grid_size, grid_size), dtype=int)

    for y in range(grid_size):
        for x in range(grid_size):
            index_matrix[y, x] = y * grid_size + x

    print("Grid Index Matrix:")
    print(index_matrix)

    # 示例：坐标到网格的映射
    example_coords = [
        (100, 200),  # 第6行第3列
        (300, 400)  # 第12行第9列
    ]

    for coord in example_coords:
        grid_x = coord[0] // grid_width
        grid_y = coord[1] // grid_width
        grid_idx = grid_y * grid_size + grid_x

        print(f"\nCoordinate {coord}:")
        print(f"Grid X: {grid_x}")
        print(f"Grid Y: {grid_y}")
        print(f"Grid Index: {grid_idx}")


# 运行可视化
visualize_grid_indexing()