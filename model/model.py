import torch
import torch.nn as nn
import os
from .VLFM import VLFM
from .Decoder import Decoder
from .MoD import MoD , Router





class Kraken_VLM(nn.Module):
    def __init__(self,vocab_size=10000):
        super().__init__()
        
        self.vlfm = VLFM()
        
        #single decoder
        #self.decoder = Decoder(vocab_size=vocab_size)
        
        #Mod
        self.decoder = MoD(vocab_size=vocab_size)
        
        #MoD router training
        #self.router = Router(n_decoders=7)
        
        checkpoint_path = os.path.join(
             os.path.dirname(__file__),
             "..",
            "VLFM.pth"
        )
        if os.path.exists(checkpoint_path):
            try:
                self.vlfm.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True))
                print("vlfm Loading saved model...")
            except RuntimeError:
                            print("vlfm Saved weights incompatible — training from scratch.")
        else :
                        print('vlfm from scratch')
                        
       

    
    def forward(self, images,
                prompt_ids ,target_ids,
                prompt_attention_mask=None ,decoder_attention_mask=None):
        
        
        vlfm_output = self.vlfm(images,prompt_ids,prompt_attention_mask)
       
        
        outputs = self.decoder(
           vlfm_output,
           target_ids,
           images,
           decoder_attention_mask,
           prompt_attention_mask,
        )
        return outputs
        
        #logits = self.router(vlfm_output)
        #return logits
       


