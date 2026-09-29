import torch
import torch.nn as nn
from .Bert import BERTEncoder
from .VisionCore import VisionCore





class VLFM(nn.Module):
    def __init__(self,d_model=768,bert_d=512,max_len=64):
        super().__init__()
        
        #text encoder
        self.text_encoder = BERTEncoder()
        #vision
        self.vision = VisionCore(
            bert_embeddings=self.text_encoder.bert.embeddings,
            bert_hidden=self.text_encoder.bert.config.hidden_size,
            d_model=d_model
        )

        self.dropout = nn.Dropout(0.1)
        
        
        
        #cat image and prompt
        self.image_type_embedding = nn.Parameter(
            torch.randn(1,1,768)
        )
        self.text_type_embedding = nn.Parameter(
            torch.randn(1,1,768)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(bert_d, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.memory_norm = nn.LayerNorm(d_model)
        #fusion encoder
        self.fusion_pos_embedding = nn.Parameter(torch.randn(1, 33 + max_len, d_model) * 0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, batch_first=True,norm_first=True 
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=6)

        self.norm = nn.LayerNorm(d_model)
        #init
        nn.init.normal_(self.image_type_embedding, std=0.02)
        nn.init.normal_(self.text_type_embedding, std=0.02)

       
    
    def forward(self, images,
                prompt_ids,
                prompt_attention_mask=None):
        
        #image proccess
        vit_output = self.vision(images, prompt_ids, prompt_attention_mask) 
        
        #prompt proccess
        prompt_output = self.text_encoder(
            prompt_ids,
            prompt_attention_mask)
        
        #cat image and prompt
        vit_output = self.image_proj(vit_output)
        vit_output = vit_output + self.image_type_embedding

        prompt_output = self.text_proj(prompt_output)
        prompt_output = prompt_output + self.text_type_embedding

        encoders_output = torch.cat([vit_output,prompt_output], dim=1)
        encoders_output = self.dropout(encoders_output)
        encoders_output = self.memory_norm( encoders_output)

        #memory mask
        image_mask = torch.zeros(
            images.size(0),
            vit_output.size(1),
            dtype=torch.bool,
            device=images.device
        )  
        #prompt_attention_mask = ~prompt_attention_mask.bool()
        memory_mask = torch.cat(
            [image_mask, prompt_attention_mask],
            dim=1
        )
        #fusion
        encoders_output = encoders_output + self.fusion_pos_embedding[:, :encoders_output.size(1)]
        output = self.fusion_encoder(
            encoders_output, src_key_padding_mask=memory_mask
        )

       
        return output
