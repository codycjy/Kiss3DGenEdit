"""
CLIPSeg Mask 生成模块

使用 CLIP 的语义理解能力，通过文本 prompt 分割图像区域
用于 P2P 编辑时精确控制编辑区域
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image


class CLIPSegMaskGenerator:
    """
    CLIPSeg 语义分割 mask 生成器

    通过文本 prompt 识别图像中的特定区域，生成二值/软 mask
    """

    def __init__(self, device: str = "cuda", model_id: str = "CIDAS/clipseg-rd64-refined"):
        """
        初始化 CLIPSeg 模型

        Args:
            device: 运行设备
            model_id: HuggingFace 模型 ID
        """
        self.device = device
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

            print(f"Loading CLIPSeg model: {self.model_id}")
            self._processor = CLIPSegProcessor.from_pretrained(self.model_id)
            self._model = CLIPSegForImageSegmentation.from_pretrained(self.model_id)
            self._model.to(self.device).eval()
            print("CLIPSeg model loaded")

    @property
    def processor(self):
        self._load_model()
        return self._processor

    @property
    def model(self):
        self._load_model()
        return self._model

    @torch.no_grad()
    def generate_mask(
        self,
        image: Union[Image.Image, torch.Tensor],
        prompt: Union[str, List[str]],
        threshold: float = 0.5,
        invert: bool = False,
        soft_mask: bool = False,
        gaussian_sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        生成语义分割 mask

        Args:
            image: 输入图像（PIL Image 或 [C, H, W] tensor）
            prompt: 分割文本（如 "face", "hair", "background"）
                    可以是多个 prompt 的列表，会合并结果
            threshold: 二值化阈值（仅 soft_mask=False 时生效）
            invert: 是否反转 mask
                    - False: prompt 区域 mask=1（保持该区域）
                    - True:  prompt 区域 mask=0（编辑该区域）
            soft_mask: 是否返回软 mask（连续值 [0,1]）
            gaussian_sigma: 高斯模糊标准差（0 表示不模糊）

        Returns:
            mask: [1, 1, H, W] tensor，值域 [0, 1]
        """
        # 转换 tensor 到 PIL
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # [1, C, H, W] -> [C, H, W]
            if image.shape[0] in [1, 3, 4]:
                # [C, H, W] -> [H, W, C]
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype("uint8")
            image = Image.fromarray(image)

        # 处理多个 prompt
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # 处理输入
        inputs = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        # 推理
        outputs = self.model(**inputs)
        logits = outputs.logits  # [num_prompts, H, W]

        # 合并多个 prompt 的结果（取 max）
        if len(prompts) > 1:
            logits = logits.max(dim=0, keepdim=True)[0]  # [1, H, W]

        # Sigmoid 转概率
        probs = torch.sigmoid(logits)  # [1, H, W]

        if soft_mask:
            mask = probs
        else:
            # 二值化
            mask = (probs > threshold).float()

        # 反转
        if invert:
            mask = 1.0 - mask

        # 添加 channel 维度
        mask = mask.unsqueeze(1)  # [1, 1, H, W]

        # 高斯模糊
        if gaussian_sigma > 0:
            mask = self._gaussian_blur(mask, sigma=gaussian_sigma)

        return mask

    def _gaussian_blur(
        self,
        x: torch.Tensor,
        kernel_size: int = 5,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        """对 tensor 应用高斯模糊"""
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 创建 1D 高斯核
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 创建 2D 高斯核
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        # 应用高斯模糊
        padding = kernel_size // 2
        x_blurred = F.conv2d(x, kernel_2d, padding=padding)

        return x_blurred

    def resize_mask_to_latent(
        self,
        mask: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """
        调整 mask 到 latent 空间尺寸

        Args:
            mask: [batch, 1, H, W]
            latent_h: latent 高度
            latent_w: latent 宽度

        Returns:
            resized mask: [batch, 1, latent_h, latent_w]
        """
        return F.interpolate(
            mask,
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        )

    def visualize_mask(
        self,
        mask: torch.Tensor,
        save_path: str,
        original_image: Optional[torch.Tensor] = None,
        overlay_alpha: float = 0.0,
    ) -> None:
        """
        可视化并保存 mask

        Args:
            mask: [1, 1, H, W] 的 mask tensor
            save_path: 保存路径
            original_image: 可选的原始图像，用于叠加显示
            overlay_alpha: 叠加透明度，0.0 表示只显示 mask
        """
        import torchvision

        # 转换为热力图
        mask_np = mask.squeeze().cpu()

        # 红-蓝热力图
        mask_rgb = torch.zeros(3, mask_np.shape[0], mask_np.shape[1])
        mask_rgb[0] = mask_np  # R channel - 高值区域
        mask_rgb[1] = mask_np * 0.2  # G channel
        mask_rgb[2] = 1 - mask_np  # B channel - 低值区域

        if overlay_alpha > 0 and original_image is not None:
            if original_image.dim() == 4:
                original_image = original_image.squeeze(0)
            if original_image.shape[-2:] != mask_rgb.shape[-2:]:
                mask_rgb = F.interpolate(
                    mask_rgb.unsqueeze(0),
                    size=original_image.shape[-2:],
                    mode="bilinear",
                ).squeeze(0)

            overlay = overlay_alpha * mask_rgb + (1 - overlay_alpha) * original_image.cpu()
            torchvision.utils.save_image(overlay, save_path)
        else:
            torchvision.utils.save_image(mask_rgb, save_path)

        print(f"CLIPSeg mask saved to: {save_path}")


def generate_clipseg_mask(
    image: Union[Image.Image, torch.Tensor],
    prompt: Union[str, List[str]],
    device: str = "cuda",
    threshold: float = 0.5,
    invert: bool = False,
    soft_mask: bool = False,
    latent_size: Optional[tuple] = None,
) -> torch.Tensor:
    """
    便捷函数：生成 CLIPSeg mask

    Args:
        image: 输入图像
        prompt: 分割文本
        device: 运行设备
        threshold: 二值化阈值
        invert: 是否反转
        soft_mask: 是否返回软 mask
        latent_size: 可选的 (h, w) 元组，自动 resize 到 latent 尺寸

    Returns:
        mask tensor
    """
    generator = CLIPSegMaskGenerator(device=device)
    mask = generator.generate_mask(
        image=image,
        prompt=prompt,
        threshold=threshold,
        invert=invert,
        soft_mask=soft_mask,
    )

    if latent_size is not None:
        mask = generator.resize_mask_to_latent(mask, latent_size[0], latent_size[1])

    return mask
