import numpy as np
import torch

from utils.util import vis_corr


def validate_single(args, model, sat, grd, sat_delta, sat_gps, meter_per_pixel, vis=False):
    model.eval()
    total_loss = 0
    all_errors = []

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    torch.cuda.empty_cache()
    with torch.no_grad():

        # 前向传播
        sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask1_dict = model(sat, grd, None,
                                                                                       None, None, meter_per_pixel)

        corr = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, None)

        # # 计算损失
        # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
        # loss = 100 * cls_loss + 1 * reg_loss

        max_level = args.levels[-1]

        B, corr_H, corr_W = corr.shape

        max_index = torch.argmax(corr.reshape(B, -1), dim=1)
        pred_u = (max_index % corr_W - corr_W / 2)
        pred_v = (max_index // corr_W - corr_H / 2)

        pred_u = pred_u * np.power(2, 3 - max_level) * meter_per_pixel
        pred_v = pred_v * np.power(2, 3 - max_level) * meter_per_pixel

        pred_us.append(pred_u.data.cpu().numpy())
        pred_vs.append(pred_v.data.cpu().numpy())

        gt_shift_u = sat_delta[:, 0] * meter_per_pixel * 512 / 4
        gt_shift_v = sat_delta[:, 1] * meter_per_pixel * 512 / 4

        gt_us.append(gt_shift_u.data.cpu().numpy())
        gt_vs.append(gt_shift_v.data.cpu().numpy())

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]
        pred_x = pred_u / meter_per_pixel + 512 / 2
        pred_y = pred_v / meter_per_pixel + 512 / 2
        if vis:
            save_path = f"./vis/distillation/{args.model_name}/test/{sat_gps[0].cpu().numpy()}.png"
            vis_corr(corr[0], sat[0], grd[0], gt_points[0], [pred_x[0], pred_y[0]], save_path)
        else:
            vis_corr(corr[0], sat[0], grd[0], gt_points[0], [pred_x[0], pred_y[0]], None)

    return


if __name__ == '__main__':
    pass