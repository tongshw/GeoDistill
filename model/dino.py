import torch

import torch.nn.functional as F
import einops as E
class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vitl14",
        output="dense-cls",
        layer=-1,
        return_multilayer=True,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vits14": 384,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        # dino_vit = torch.hub.load('./dinov2', 'dinov2_vitl14', weights={'LVD142M':'./dinov2_models/dinov2_vitl14_pretrain.pth'}, source='local')
        self.vit = dino_vit.eval().to(torch.float32)

        # 冻结参数
        for param in self.vit.parameters():
            param.requires_grad = False

        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs

def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output