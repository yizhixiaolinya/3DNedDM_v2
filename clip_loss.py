import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from CLIP.model import CLIP
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

        MODEL_CONFIG_PATH = 'CLIP/model_config.yaml'
        model_config = load_config_file(MODEL_CONFIG_PATH)

        model_params = dict(model_config.RN50)
        model_params['vision_layers'] = tuple(model_params['vision_layers'])
        model_params['vision_patch_size'] = None
        # 初始化模型
        self.model = CLIP(**model_params).to(self.device)

        # 如果有预训练权重，可以在此加载
        state_dict = torch.load('/public_bme/home/linxin/13119176/checkpoint_CLIP.pt', map_location=self.device)
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

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        images = images.float()

        # 将图像归一化到 [0, 1]
        images = (images - images.min()) / (images.max() - images.min())

        # 将图像的类型转换为模型的权重类型
        images = images.type(self.model.dtype)

        # 编码图像
        return self.model.encode_image(images)

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.encode_images(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encodings = self.get_image_features(target_images)
            target_encodings = target_encodings.mean(dim=0, keepdim=True)

            direction = target_encodings - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def set_text_features(self, source_class_features: torch.Tensor, target_class_features: torch.Tensor) -> None:
        source_features = source_class_features.squeeze(1)  # (batch_size, 768)
        source_features = source_features.mean(dim=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = target_class_features.squeeze(1)  # (batch_size, 768)
        target_features = target_features.mean(dim=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, source_class: torch.Tensor, target_img: torch.Tensor,
                        target_class: torch.Tensor) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = (self.target_text_features @ self.src_text_features.T).clamp(-1, 1)
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img)
        target_img_features = self.get_image_features(target_img)

        cos_img_angle = (target_img_features * src_img_features).sum(dim=-1, keepdim=True).clamp(-1, 1)
        img_angle = torch.acos(cos_img_angle)

        return self.angle_loss(img_angle, text_angle)

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: torch.Tensor, target_img: torch.Tensor,
                              target_class: torch.Tensor) -> torch.Tensor:
        if self.target_direction is None:
            self.target_direction = (target_class - source_class).mean(dim=0, keepdim=True)
            self.target_direction /= self.target_direction.norm(dim=-1, keepdim=True)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= (edit_direction.norm(dim=-1, keepdim=True) + 1e-7)
        return self.direction_loss(edit_direction, self.target_direction).mean()

    def global_clip_loss(self, img: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        tokens = text_features.squeeze(1)  # (batch_size, 768)
        tokens = tokens.to(self.device)
        tokens /= tokens.norm(dim=-1, keepdim=True)

        images = img.to(self.device)
        images = images.float()
        images = (images - images.min()) / (images.max() - images.min())

        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ tokens.T

        return (1. - logits_per_image / 100).mean()

    def forward(self, src_img: torch.Tensor, source_class: torch.Tensor, target_img: torch.Tensor,
                target_class: torch.Tensor):
        clip_loss = 0.0

        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(target_img, target_class)

        if self.lambda_direction:
            clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img,
                                                                            target_class)

        if self.lambda_manifold:
            clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)

        return clip_loss
