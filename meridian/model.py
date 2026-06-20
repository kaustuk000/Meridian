from __future__ import annotations
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import os

from transformers import CLIPModel, CLIPProcessor
from meridian.lorentz import exp_map0, log_map0,lorentz_inner_product

class LayerAggregator(nn.Module):
    def __init__(self, num_layers: int = 12):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states):
        """
        hidden_states: tuple of length 13 containing:
                       hidden_states[0] = embedding layer output
                       hidden_states[1] to [12] = transformer block outputs 
        """
        # Exclude the raw embedding layer (index 0) and stack the 12 transformer block outputs
        # Stack shape: [12, Batch, Seq_Len, Hidden_Dim]
        layers = torch.stack(hidden_states[1:], dim = 0)

        # Turn weights into a probability distribution that sums to 1
        normalized_weights = torch.softmax(self.weights, dim = 0)

        # Reshape weights for broadcasting: [12, 1, 1, 1]
        normalized_weights = normalized_weights.view(-1, 1, 1, 1)

        # Multiply each layer by its weight and sum them together
        aggregated_tokens = (layers * normalized_weights).sum(dim = 0)

        return aggregated_tokens


class QuickGelu(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)

class TransformerAdapter(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, ff_mult: int = 4,
        dropout: float = 0.1, num_layers: int = 2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = input_dim,
            nhead = num_heads,
            dim_feedforward = input_dim * ff_mult,
            dropout = dropout,
            batch_first = True,
            norm_first = True,
            activation = QuickGelu(),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm = nn.LayerNorm(input_dim),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None):
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

class HyperbolicImageHead(nn.Module):
    """
    Image:
        patch tokens
            -> adapter
            -> CLS token
            -> projection
            -> scaling
            -> expmap0
    """

    def __init__(self, input_dim: int = 768, out_dim: int = 64, adapter_layers: int = 2):
        super().__init__()
        self.image_adapter = TransformerAdapter(input_dim, num_layers = adapter_layers)
        self.image_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            QuickGelu(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_dim),
        )

        for m in self.image_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.log_alpha_img = nn.Parameter(torch.tensor(out_dim**-0.5).log())  # Initialize alpha to a small value
        
    def forward(self, image_tokens: Tensor, key_padding_mask: Tensor | None = None, curv: float | Tensor = 1.0) -> Tensor:
        image_tokens = self.image_adapter(image_tokens, key_padding_mask=key_padding_mask)
        cls_token = image_tokens[:,0] # Extract CLS token
        v = self.image_mlp(cls_token)

        with torch.autocast(device_type=v.device.type, dtype=torch.float32):
            alpha = self.log_alpha_img.exp()
            v = alpha * v
            h_image = exp_map0(v, curv = curv) # Map to hyperbolic space using expmap0
        return h_image

class HyperbolicTextHead(nn.Module):
    """
    Text:
        text tokens
            -> adapter
            -> EOS token
            -> projection
            -> scaling
            -> expmap0
    """

    def __init__(self, input_dim: int = 512, out_dim: int = 64, adapter_layers: int = 2):
        super().__init__()
        self.text_adapter = TransformerAdapter(input_dim, num_layers = adapter_layers)
        self.text_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            QuickGelu(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_dim),
        )

        for m in self.text_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.log_alpha_txt = nn.Parameter(torch.tensor(out_dim**-0.5).log())  # Initialize alpha to a small value

    def forward(self, text_tokens: Tensor, eos_indices: Tensor, attention_mask: Tensor | None = None, curv: float | Tensor = 1.0) -> Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        text_tokens = self.text_adapter(text_tokens, key_padding_mask=key_padding_mask)

        eos_indices = eos_indices.to(text_tokens.device)
        batch_idx = torch.arange(text_tokens.size(0), device=text_tokens.device)
        eos_token = text_tokens[batch_idx, eos_indices]

        v = self.text_mlp(eos_token)

        with torch.autocast(device_type=v.device.type, dtype=torch.float32):
            alpha = self.log_alpha_txt.exp()
            v = alpha * v
            h_text = exp_map0(v, curv = curv) # Map to hyperbolic space using expmap0
        return h_text

class EuclideanImageHead(nn.Module):
    def __init__(self, input_dim: int = 768, out_dim: int = 64, adapter_layers: int = 2):
        super().__init__()
        self.image_adapter = TransformerAdapter(input_dim, num_layers = adapter_layers)
        self.image_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            QuickGelu(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_dim),
        )

        for m in self.image_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, image_tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        image_tokens = self.image_adapter(image_tokens, key_padding_mask=key_padding_mask)
        cls_token = image_tokens[:,0] # Extract CLS token
        v = self.image_mlp(cls_token)
        e_image = F.normalize(v,dim = -1, eps=1e-6)
        return e_image

class EuclideanTextHead(nn.Module):
    def __init__(self, input_dim: int = 512, out_dim: int = 64, adapter_layers: int = 2):
        super().__init__()
        self.text_adapter = TransformerAdapter(input_dim, num_layers = adapter_layers)
        self.text_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            QuickGelu(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_dim),
        )

        for m in self.text_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_tokens: Tensor, eos_indices: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        text_tokens = self.text_adapter(text_tokens, key_padding_mask=key_padding_mask)

        eos_indices = eos_indices.to(text_tokens.device)
        batch_idx = torch.arange(text_tokens.size(0), device=text_tokens.device)
        eos_token = text_tokens[batch_idx, eos_indices]

        v = self.text_mlp(eos_token)
        e_text = F.normalize(v, dim = -1, eps=1e-6)
        return e_text

class PerModalityGate(nn.Module):
    def __init__(self, hyp_dim: int = 64, eucl_dim: int = 64, clip_dim: int = 512):
        super().__init__()
        # Maps combined features to 2 raw logits [hyperbolic_logit, euclidean_logit]
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            QuickGelu(),
            nn.Linear(128, hyp_dim),       
        )

        for m in self.clip_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.linear = nn.Sequential(
            nn.Linear(hyp_dim + eucl_dim + hyp_dim + 1, 16),
            QuickGelu(),
            nn.Linear(16, 2),
        )
        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, hyp: Tensor, eucl: Tensor, clip_feat: Tensor, curv: float | Tensor = 1.0) -> Tensor:

        with torch.autocast(device_type=hyp.device.type, dtype=torch.float32):
            # Project hyperbolic vector back to tangent space at origin
            flat_h = log_map0(hyp, curv = curv)
            h_radius = flat_h.norm(dim=-1, keepdim=True)
            h_dir = F.normalize(flat_h, dim=-1, eps=1e-6)
            e_dir = eucl # Euclidean vectors are already normalized
            clip_dir = F.normalize(self.clip_proj(clip_feat), dim=-1, eps=1e-6)

            # Concatenate vectors
            combined = torch.cat([h_dir, e_dir, clip_dir,h_radius], dim=-1)
            logits = self.linear(combined)

        # Return raw logits
        return logits



class MeridianModel(nn.Module):
    def __init__(self, image_hout: int = 16, image_eout: int = 16, text_hout: int = 16, text_eout: int = 16, curv_init: float = 1.0, 
        learn_curv: bool = True):
        super().__init__()

        # Load Hugging Face Model and Processor
        local_model_path = r"checkpoints/clip/hf_vitb16_openai" 
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model path {local_model_path} does not exist. Please run scripts/download_models.py first.")
        print(f"Loading Hugging Face model from {local_model_path}...")

        self.clip = CLIPModel.from_pretrained(local_model_path)
        self.processor = CLIPProcessor.from_pretrained(local_model_path)

        # Freeze the CLIP model parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        self.clip.eval()  # Put in eval mode to freeze dropout/batchnorm behavior
        
        # Selective unfreeze tracking
        # Populated by unfreeze_last_n_layer() drives the train() override so
        # that model.train() doesn't accidentally re-freeze these blocks
        self._unfrozen_clip_modules: list[nn.Module] = []


        # Curvature parameter
        if learn_curv:  
            self.curv = nn.Parameter(torch.tensor(curv_init).log())  # Learnable curvature
        else:
            self.register_buffer('curv', torch.tensor(curv_init))  # Fixed curvature

        self._curv_minmax = {
            "max": math.log(curv_init * 10),  
            "min": math.log(curv_init / 8), 
        }
        
        # Temperature for Contrastive Loss euclidean
        self.logit_scale_eucl = nn.Parameter(torch.tensor(1.0 / 0.07).log())
        # Temperature for Contrastive Loss hyperbolic
        self.logit_scale_hyp = nn.Parameter(torch.tensor(1.0 / 0.07).log())

        # Initialize the Heads and Aggregators
        print("Initializing LayerAggregators and Hyperbolic/Euclidean Heads...")

        # Initialize the Aggregators
        self.hyp_image_aggregator = LayerAggregator(num_layers = 12)
        self.hyp_text_aggregator = LayerAggregator(num_layers = 12)

        self.eucl_image_aggregator = LayerAggregator(num_layers = 12)
        self.eucl_text_aggregator = LayerAggregator(num_layers = 12)

        # Initialize the Hyperbolic/Euclidean Heads
        self.hyp_image_head = HyperbolicImageHead(input_dim = 768, out_dim = image_hout)
        self.hyp_text_head = HyperbolicTextHead(input_dim = 512, out_dim = text_hout)

        self.eucl_image_head = EuclideanImageHead(input_dim = 768, out_dim = image_eout)
        self.eucl_text_head = EuclideanTextHead(input_dim = 512, out_dim = text_eout)
        
        # Initialize the PerModalityGate
        self.gate_image = PerModalityGate(hyp_dim = image_hout, eucl_dim = image_eout, clip_dim = 768)
        self.gate_text  = PerModalityGate(hyp_dim = text_hout, eucl_dim = text_eout, clip_dim = 512)


 # ── Selective CLIP unfreeze ────────────────────────────────────────────────
 
    def unfreeze_last_n_layers(self, n: int) -> list[nn.Parameter]:
        """
        Unfreeze the last *n* transformer blocks in both the CLIP vision encoder
        and text encoder, plus their trailing LayerNorms.
 
        Args:
            n: Number of transformer blocks to unfreeze (counted from the top).
               Pass 0 to keep everything frozen (no-op).
 
        Returns:
            List of newly requires_grad=True parameters, ready to be handed
            to optimizer.add_param_group().
        """
        if n <= 0:
            return []
 
        vision_layers = list(self.clip.vision_model.encoder.layers)
        text_layers   = list(self.clip.text_model.encoder.layers)
        total_v = len(vision_layers)   # 12 for ViT-B/16
        total_t = len(text_layers)     # 12 for ViT-B/16
 
        if n > total_v:
            raise ValueError(
                f"Requested {n} vision layers but CLIP only has {total_v}."
            )
        if n > total_t:
            raise ValueError(
                f"Requested {n} text layers but CLIP only has {total_t}."
            )
 
        modules_to_unfreeze: list[nn.Module] = (
            vision_layers[-n:]
            + text_layers[-n:]
        )
 
        new_params: list[nn.Parameter] = []
        for module in modules_to_unfreeze:
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    new_params.append(param)
 
        self._unfrozen_clip_modules = modules_to_unfreeze
 
        # If we're already in train mode, put these submodules into train mode
        # immediately (train() won't be called again until the next epoch boundary).
        if self.training:
            for module in self._unfrozen_clip_modules:
                module.train()
 
        return new_params
 
    # ── train() override ──────────────────────────────────────────────────────
 
    def train(self, mode: bool = True):
        """
        Keep the frozen CLIP backbone in eval mode at all times.
        Selectively unfrozen modules (populated by unfreeze_last_n_layers)
        are put back into the requested mode so their dropout/BN behave
        correctly during fine-tuning.
        """
        super().train(mode)
        # Force the entire CLIP backbone into eval first …
        self.clip.eval()
        # … then re-enable train mode only for the explicitly unfrozen blocks.
        for module in self._unfrozen_clip_modules:
            module.train(mode)
        return self


    def encode_image(self, pixel_values: Tensor) -> dict:
        clamped_curv_log = torch.clamp(self.curv, min=self._curv_minmax["min"], max=self._curv_minmax["max"])
        _curv = clamped_curv_log.exp()
        hyp_temp = torch.clamp(self.logit_scale_hyp, max=4.6052).exp()
        eucl_temp = torch.clamp(self.logit_scale_eucl, max=4.6052).exp()

        #with torch.no_grad():
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        image_hidden_states = vision_outputs.hidden_states

        img_clip_feat = image_hidden_states[-1][:, 0]

        hyp_image_tokens = self.hyp_image_aggregator(image_hidden_states)
        eucl_image_tokens = self.eucl_image_aggregator(image_hidden_states)

        h_image = self.hyp_image_head(image_tokens=hyp_image_tokens, key_padding_mask=None, curv=_curv)
        e_image = self.eucl_image_head(image_tokens=eucl_image_tokens, key_padding_mask=None)

        logits_img = self.gate_image(h_image, e_image, img_clip_feat, curv=_curv)
        a_img, b_img = F.softmax(logits_img, dim=-1).unbind(dim=-1)

        return {"h_image": h_image, "e_image": e_image, "a_img": a_img, "b_img": b_img,
                "curv": _curv, "scale_hyp": hyp_temp, "scale_eucl": eucl_temp}


    def encode_text(self, input_ids: Tensor, attention_mask: Tensor, eos_indices: Tensor) -> dict:
        clamped_curv_log = torch.clamp(self.curv, min=self._curv_minmax["min"], max=self._curv_minmax["max"])
        _curv = clamped_curv_log.exp()
        hyp_temp = torch.clamp(self.logit_scale_hyp, max=4.6052).exp()
        eucl_temp = torch.clamp(self.logit_scale_eucl, max=4.6052).exp()

        #with torch.no_grad():
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        text_hidden_states = text_outputs.hidden_states

        eos_indices = eos_indices.to(text_hidden_states[-1].device)
        batch_idx = torch.arange(text_hidden_states[-1].size(0), device=text_hidden_states[-1].device)
        txt_clip_feat = text_hidden_states[-1][batch_idx, eos_indices]

        hyp_text_tokens = self.hyp_text_aggregator(text_hidden_states)
        eucl_text_tokens = self.eucl_text_aggregator(text_hidden_states)

        h_text = self.hyp_text_head(text_tokens=hyp_text_tokens, eos_indices=eos_indices, attention_mask=attention_mask, curv=_curv)
        e_text = self.eucl_text_head(text_tokens=eucl_text_tokens, eos_indices=eos_indices, attention_mask=attention_mask)

        logits_txt = self.gate_text(h_text, e_text, txt_clip_feat, curv=_curv)
        a_txt, b_txt = F.softmax(logits_txt, dim=-1).unbind(dim=-1)

        return {"h_text": h_text, "e_text": e_text, "a_txt": a_txt, "b_txt": b_txt,
                "curv": _curv, "scale_hyp": hyp_temp, "scale_eucl": eucl_temp}


    def forward(self, pixel_values: Tensor, input_ids: Tensor, attention_mask: Tensor, eos_indices: Tensor):
        """
        Args:
            pixel_values: (Batch, 3, 224, 224)
            input_ids: (Batch, Seq_Len)
            attention_mask: (Batch, Seq_Len)
            eos_indices: (Batch,) - The index of the EOS token for each sequence   
        """
        image_feats = self.encode_image(pixel_values)
        text_feats = self.encode_text(input_ids, attention_mask, eos_indices)

        h_image = image_feats["h_image"]
        e_image = image_feats["e_image"]
        a_img = image_feats["a_img"]
        b_img = image_feats["b_img"]

        h_text = text_feats["h_text"]
        e_text = text_feats["e_text"]
        a_txt = text_feats["a_txt"]
        b_txt = text_feats["b_txt"]

        curv = image_feats["curv"]
        scale_hyp = image_feats["scale_hyp"]
        scale_eucl = image_feats["scale_eucl"]

        return {
            "h_image": h_image,
            "h_text": h_text,
            "e_image": e_image,
            "e_text": e_text,
            "a_image": a_img,
            "b_image": b_img,
            "a_text": a_txt,
            "b_text": b_txt,
            "curv": curv,
            "scale_eucl": scale_eucl,
            "scale_hyp": scale_hyp,
        }
