from __future__ import annotations

import cv2

import functools
import time
import math

import numpy as np
import torch
import torch.nn.functional as F

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)



def get_perspective_transform(src, dst):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not torch.is_tensor(src):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(dst):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(src.shape, dst.shape))

    def ax(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], dim=1)

    def ay(p, q):
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        return torch.cat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)
    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    p.append(ax(src[:, 0], dst[:, 0]))
    p.append(ay(src[:, 0], dst[:, 0]))

    p.append(ax(src[:, 1], dst[:, 1]))
    p.append(ay(src[:, 1], dst[:, 1]))

    p.append(ax(src[:, 2], dst[:, 2]))
    p.append(ay(src[:, 2], dst[:, 2]))

    p.append(ax(src[:, 3], dst[:, 3]))
    p.append(ay(src[:, 3], dst[:, 3]))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], dim=1)

    # solve the system Ax = b
    # X, LU = torch.gesv(b, A)
    X = torch.linalg.solve(A, b)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)
    return M.view(-1, 3, 3)  # Bx3x3

def gps2distance(Lat_A, Lng_A, Lat_B, Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    device = torch.device('cuda:0')
    lat_A = torch.deg2rad(Lat_A.double())  # Lat_A * torch.pi/180.
    lat_B = torch.deg2rad(Lat_B.double())
    lng_A = torch.deg2rad(Lng_A.double())
    lng_B = torch.deg2rad(Lng_B.double())
    R = torch.tensor(6371004.).cuda(device)  # Earth's radius in meters
    C = torch.sin(lat_A) * torch.sin(lat_B) + torch.cos(lat_A) * torch.cos(lat_B) * torch.cos(lng_A - lng_B)
    C = torch.clamp(C, min=-1.0, max=1.0)
    distance = R * torch.acos(C)
    return distance


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the real distance between two locations based on GPS latitude and longitude values.
    :param lat1: Latitude of location 1
    :param lon1: Longitude of location 1
    :param lat2: Latitude of location 2
    :param lon2: Longitude of location 2
    :return: The real distance between the two locations (in meters)
    """
    # Convert latitude and longitude to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Use the Haversine formula to calculate the distance between two locations
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371393  # Earth's radius (in meters)
    distance = c * r

    return distance


def get_BEV_projection(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, device='cpu'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device

    Hp, Wp = img.shape[0], img.shape[1]  # Panorama image dimensions

    Fov = Fov * torch.pi / 180  # Field of View in radians
    center = torch.tensor([Wp / 2 + dx, Hp + dy]).to(device)  # Overhead view center

    anglex = torch.tensor(dx).to(device) * 2 * torch.pi / Wp
    angley = -torch.tensor(dy).to(device) * torch.pi / Hp
    anglez = torch.tensor(0).to(device)

    # Euler angles
    euler_angles = (anglex, angley, anglez)
    euler_angles = torch.stack(euler_angles, -1)

    # Calculate the rotation matrix
    R02 = euler_angles_to_matrix(euler_angles, "XYZ")
    R20 = torch.inverse(R02)

    f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
    out = torch.zeros((Wo, Ho, 2)).to(device)
    f0 = torch.zeros((Wo, Ho, 3)).to(device)
    f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T
    f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device)
    f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
    f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
    f1_0 = torch.sqrt(torch.sum(f1 ** 2, 0))
    f1_1 = torch.sqrt(torch.sum(f1[:2, :] ** 2, 0))
    theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
    phi = torch.arctan2(f1[1, :], f1[0, :])  # [-pi, pi]
    phi = phi + torch.pi  # [0, 2pi]

    i_p = 1 - theta / torch.pi  # [0, 1]
    j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
    out[:, :, 0] = j_p.reshape((Ho, Wo))
    out[:, :, 1] = i_p.reshape((Ho, Wo))
    out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
    out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]

    return out


def get_BEV_tensor(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, dataset=False, out=None, device='cpu'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device

    t0 = time.time()
    Hp, Wp = img.shape[0], img.shape[1]  # Panorama image dimensions
    if dty != 0 or Wp != 2 * Hp:
        ty = (Wp / 2 - Hp) / 2 + dty  # Non-standard panorama image completion
        matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
        img = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))
    ######################
    t1 = time.time()
    # frame = torch.from_numpy(img.astype(np.float32)).to(device)
    frame = torch.from_numpy(img.copy()).to(device)
    t2 = time.time()

    if out is None:
        Fov = Fov * torch.pi / 180  # Field of View in radians
        center = torch.tensor([Wp / 2 + dx, Hp + dy]).to(device)  # Overhead view center

        anglex = torch.tensor(dx).to(device) * 2 * torch.pi / Wp
        angley = -torch.tensor(dy).to(device) * torch.pi / Hp
        anglez = torch.tensor(0).to(device)

        # Euler angles
        euler_angles = (anglex, angley, anglez)
        euler_angles = torch.stack(euler_angles, -1)

        # Calculate the rotation matrix
        R02 = euler_angles_to_matrix(euler_angles, "XYZ")
        R20 = torch.inverse(R02)

        f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
        out = torch.zeros((Wo, Ho, 2)).to(device)
        f0 = torch.zeros((Wo, Ho, 3)).to(device)
        f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T
        f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device)
        f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
        f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
        # f1 = f0.reshape((-1, 3)).T
        f1_0 = torch.sqrt(torch.sum(f1 ** 2, 0))
        f1_1 = torch.sqrt(torch.sum(f1[:2, :] ** 2, 0))
        theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
        phi = torch.arctan2(f1[1, :], f1[0, :])  # [-pi, pi]
        phi = phi + torch.pi  # [0, 2pi]

        i_p = 1 - theta / torch.pi  # [0, 1]
        j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
        out[:, :, 0] = j_p.reshape((Ho, Wo))
        out[:, :, 1] = i_p.reshape((Ho, Wo))
        out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
        out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]
    # else:
    #     out = out.to(device)
    t3 = time.time()

    BEV = F.grid_sample(frame.permute(2, 0, 1).unsqueeze(0).float(), out.unsqueeze(0), align_corners=True)
    t4 = time.time()
    # print("Read image ues {:.2f} ms, warpPerspective image use {:.2f} ms, Get matrix ues {:.2f} ms, Get out ues {:.2f} ms, All out ues {:.2f} ms.".format((t1-t0)*1000,(t2-t1)*1000, (t3-t2)*1000,(t4-t3)*1000,(t4-t0)*1000))
    if dataset:
        return BEV.squeeze(0)
    else:
        return BEV.permute(0, 2, 3, 1).squeeze(0).int()
