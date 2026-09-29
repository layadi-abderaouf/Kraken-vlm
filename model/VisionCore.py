import torch
import torch.nn as nn
import os
from .CLIP import CLIP_encoder
from .QFormer import QFormer



class VisionCore(nn.Module):

    def __init__(self,bert_embeddings, bert_hidden=512,d_model=768):
        super().__init__()
        self.clip = CLIP_encoder()
        self.q_former = QFormer()
        
        self.bert_embeddings = bert_embeddings
        self.text_proj_in = (
            nn.Linear(bert_hidden, d_model) if bert_hidden != d_model else nn.Identity()
        )
        
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "Qformer.pth"
        )
        if os.path.exists(checkpoint_path):
                try:
                    self.q_former.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True))
                    print("sformer Loading saved model...")
                except RuntimeError:
                    print("qformer Saved weights incompatible — training from scratch.")
        else :
                print('qformer from scratch')
        
        
    def build_instructblip_mask(self,num_queries, text_len, device):
       total = num_queries + text_len
       mask = torch.zeros(total, total, device=device)
       # Text لا يرى Queries
       mask[num_queries:, :num_queries] = float('-inf')
       return mask

    def forward(self, images,instruction_ids, instruction_attention_mask ):
       
        image_features = self.clip(images)
        instr_embeds = self.text_proj_in(
            self.bert_embeddings(input_ids=instruction_ids)
        )
        text_len = instr_embeds.size(1)
        attn_mask = self.build_instructblip_mask(
            num_queries=32, 
            text_len=text_len,
            device=images.device
        )
        
        B = instruction_attention_mask.size(0)
        query_padding = torch.zeros(B, self.q_former.num_queries, dtype=torch.bool, device=images.device)
        instr_padding = (instruction_attention_mask == 0)
        combined_padding = torch.cat([query_padding, instr_padding], dim=1)

        query_out = self.q_former(
            image_features, instr_embeds,
            self_attn_mask=attn_mask,
            text_key_padding_mask=combined_padding,
            return_full_output=False
        )
       
        return query_out








        