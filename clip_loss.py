import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from CLIP.model import CLIP
from utils_clip.simple_tokenizer import SimpleTokenizer
import yaml
from utils_clip import load_config_file

class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_func = {
            'mse':    torch.nn.MSELoss(),
            'cosine': torch.nn.CosineSimilarity(dim=1),
            'mae':    torch.nn.L1Loss()
        }[loss_type]

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_patch=0., lambda_global=0., lambda_manifold=0.,
                 lambda_texture=0., patch_loss_type='mae', direction_loss_type='cosine', clip_model=None):
        super(CLIPLoss, self).__init__()

        self.device = device

        checkpoint_path = '/public_bme/home/linxin/13119176/checkpoint_CLIP.pt'
        MODEL_CONFIG_PATH = 'CLIP/model_config.yaml'
        model_config = load_config_file(MODEL_CONFIG_PATH)

        self.tokenizer = SimpleTokenizer()
        model_params = dict(model_config.RN50)
        model_params['vision_layers'] = tuple(model_params['vision_layers'])
        model_params['vision_patch_size'] = None

        # 初始化模型并移动到设备上
        self.model = CLIP(**model_params).to(self.device)

        # 加载预训练的模型权重，确保权重也在正确的设备上
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)

        self.target_direction = None
        self.patch_text_directions = None

        self.patch_loss = DirectionLoss(patch_loss_type)
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)

        self.lambda_global = lambda_global
        self.lambda_patch = lambda_patch
        self.lambda_direction = lambda_direction
        self.lambda_manifold = lambda_manifold
        self.lambda_texture = lambda_texture

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        # 假设 text_features 的形状为 (batch_size, 1, 768)
        text_features = text_features.squeeze(1)  # 变为 (batch_size, 768)
        return text_features.to(self.device)

    def crop_img(self,images,crop_size):
        H, W, D = images.shape[1], images.shape[2], images.shape[3]

        start_h = (H - crop_size) // 2
        start_w = (W - crop_size) // 2
        start_d = (D - crop_size) // 2

        # Perform the cropping by slicing
        cropped_images = images[:,
                         start_h:start_h + crop_size,
                         start_w:start_w + crop_size,
                         start_d:start_d + crop_size]
        return cropped_images 

    def resize_img(self, x, length=96):
        # 调整影像的大小为 (1, 96, 96, 96)
        d0 = length - x.shape[1]
        d1 = length - x.shape[2]
        d2 = length - x.shape[3]

        if d0 < 0:
            x = x[:-abs(d0)]  # crop
        elif d0 > 0:
            padding0 = torch.zeros(d0, x.shape[1], x.shape[2])
            x = torch.cat([x, padding0], dim=0)  # padding

        if d1 < 0:
            x = x[:, :-abs(d1), :]
        elif d1 > 0:
            padding1 = torch.zeros(x.shape[0], d1, x.shape[2])
            x = torch.cat([x, padding1], dim=1)

        if d2 < 0:
            x = x[:, :, :-abs(d2)]
        elif d2 > 0:
            padding2 = torch.zeros(x.shape[0], x.shape[1], d2)
            x = torch.cat([x, padding2], dim=2)

        return x

    def get_image_features(self, img: torch.Tensor) -> torch.Tensor:
        # 确保图像的形状和数据类型正确
        images = img.to(self.device).float()
        # 编码图像特征
        images = images.reshape(images.shape[-4], images.shape[-3], images.shape[-2], images.shape[-1])
        images = self.crop_img(images, 96)
        images = images.unsqueeze(1)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def set_text_features(self, src_seq_features: torch.Tensor, tgt_seq_features: torch.Tensor) -> None:
        source_features = src_seq_features.squeeze(1)  # (batch_size, 768)
        source_features = source_features.mean(dim=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = tgt_seq_features.squeeze(1)  # (batch_size, 768)
        target_features = target_features.mean(dim=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, src_seq: torch.Tensor, tgt_img: torch.Tensor,
                        tgt_seq: torch.Tensor) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(src_seq, tgt_seq)

        cos_text_angle = (self.target_text_features @ self.src_text_features.T)
        cos_text_angle = cos_text_angle.clamp(-1, 1)
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img)
        tgt_img_features = self.get_image_features(tgt_img)

        cos_img_angle = (tgt_img_features * src_img_features).sum(dim=-1, keepdim=True)
        cos_img_angle = cos_img_angle.clamp(-1, 1)
        img_angle = torch.acos(cos_img_angle)

        return self.angle_loss(img_angle, text_angle)

    def clip_directional_loss(self, src_img: torch.Tensor, src_seq: torch.Tensor, tgt_img: torch.Tensor,
                              tgt_seq: torch.Tensor) -> torch.Tensor:
        src_seq = self.encode_text(src_seq) # 将(b,1,768)变为(b,768)
        tgt_seq = self.encode_text(tgt_seq)
        if self.target_direction is None:
            self.target_direction = (tgt_seq - src_seq).mean(dim=0, keepdim=True)
            self.target_direction = self.target_direction / self.target_direction.norm(dim=-1, keepdim=True)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(tgt_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction = edit_direction / (edit_direction.norm(dim=-1, keepdim=True) + 1e-7)
        return self.direction_loss(edit_direction, self.target_direction).mean()

    def global_clip_loss(self, img: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        # 确保通道维度为1
        tgt_seq = tgt_seq.to(self.device)
        tgt_seq = tgt_seq / tgt_seq.norm(dim=-1, keepdim=True)

        image_features = self.get_image_features(img)

        #logits_per_image = image_features @ tokens.T#TODO 乖乖 这里你明早一定要看一下 BERT处理text的步骤 token并不是sentence的feature
        print(image_features.shape,'image features')
        tgt_seq = tgt_seq.reshape(tgt_seq.shape[0],tgt_seq.shape[2])
        logits_per_image = image_features @ tgt_seq.T
        return (1. - logits_per_image / 100).mean()

    def forward(self, src_img: torch.Tensor, src_seq: torch.Tensor, tgt_img: torch.Tensor,
                tgt_seq: torch.Tensor):
        clip_loss = 0.0

        print(f"img shape before loss: {src_img.shape}")  # ([2, 1, 144, 192, 192])
        print(f"sqe shape before loss: {src_seq.shape}")

        # 比较图像和文本的特征向量，计算它们之间的相似度
        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(tgt_img, tgt_seq)
        print('global loss is done')

        # 计算源图像和目标图像的特征向量之间的方向差异，并与文本方向进行比较。
        if self.lambda_direction:
            clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, src_seq, tgt_img, tgt_seq)
        print('direction loss is done')

        # 计算源图像和目标图像的特征向量之间的角度差异，并与文本角度进行比较。
        if self.lambda_manifold:
            clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, src_seq, tgt_img, tgt_seq)
        print('manifold loss is done')

        return clip_loss
