import torch
import torch.nn.functional as F
from transformers import CLIPTokenizerFast, CLIPModel

from config import (
    CLIP_MODEL_NAME, CLIP_RENDER_MODE, STATE_CLIP_VALUE,
    CLIP_MEAN, CLIP_STD,
)


class CLIPLoss:
    def __init__(self):
        import config as _cfg

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self._std  = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        # Read prompts after CLI overrides have been applied in main.py.
        self.objective_prompts = list(_cfg.OBJECTIVE_PROMPTS)

    def _device(self):
        return next(self.model.parameters()).device

    def _finite_clip_state(self, x):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return x.clamp(-STATE_CLIP_VALUE, STATE_CLIP_VALUE)

    @staticmethod
    def _stable_l2_normalize(x, dim=-1, eps=1e-6):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        norm = x.pow(2).sum(dim=dim, keepdim=True).add(eps).sqrt()
        return x / norm

    def render_rgb(self, x):
        # x: [B, C, H, W] NCHW
        x = self._finite_clip_state(x)
        if CLIP_RENDER_MODE == 'hard':
            rgb   = x[:, :3].clamp(0.0, 1.0)
            alpha = x[:, 3:4].clamp(0.0, 1.0)
        else:
            rgb   = torch.sigmoid(x[:, :3])
            alpha = torch.sigmoid(5.0 * (x[:, 3:4] - 0.5))
        return 1.0 - alpha + alpha * rgb

    def preprocess(self, x):
        """Convert an NCHW NCA state into normalized CLIP image input."""
        rgb = self.render_rgb(x)
        rgb = torch.where(torch.isfinite(rgb), rgb, torch.ones_like(rgb))
        rgb = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)
        mean = self._mean.to(rgb)
        std  = self._std.to(rgb)
        return (rgb - mean) / std

    def embed_text(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77,
        )
        tokens = {k: v.to(self._device()) for k, v in tokens.items()}
        z = self.model.get_text_features(**tokens).float()
        return self._stable_l2_normalize(z, dim=-1)[0]

    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self.preprocess(x)
        z = self.model.get_image_features(pixel_values=pixel_values).float()
        return self._stable_l2_normalize(z, dim=-1)

    def embed_objective_prompts(self, prompts=None):
        prompts = self.objective_prompts if prompts is None else list(prompts)
        with torch.no_grad():
            return [self.embed_text(p) for p in prompts]

    def compute_objective_losses(self, x: torch.Tensor, text_embeddings) -> list:
        z_img = self.embed_image(x)
        losses = []
        for z_text in text_embeddings:
            sim = (z_img * z_text.unsqueeze(0)).sum(dim=-1)
            sim = torch.where(torch.isfinite(sim), sim, torch.zeros_like(sim))
            losses.append(-sim)
        return losses

    def loss(self, x: torch.Tensor, z_text1: torch.Tensor, z_text2: torch.Tensor):
        losses = self.compute_objective_losses(x, [z_text1, z_text2])
        return losses[0], losses[1]
