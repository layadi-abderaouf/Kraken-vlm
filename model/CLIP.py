import torch
import torch.nn as nn
from transformers import CLIPVisionModel,logging
from transformers import CLIPImageProcessor
import config

class CLIP_encoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        logging.set_verbosity_error()

        self.clip = CLIPVisionModel.from_pretrained(
            config.CLIP_MODEL
        )
        processor = CLIPImageProcessor.from_pretrained(
           config.CLIP_MODEL
        )

        # تجميد CLIP
        for p in self.clip.parameters():
            p.requires_grad = False

        self.clip.eval()

    def forward(self, x):

        with torch.no_grad():

            outputs = self.clip(
                pixel_values=x
            )

            # last_hidden_state
            # [B, 197, 768]
            return outputs.last_hidden_state