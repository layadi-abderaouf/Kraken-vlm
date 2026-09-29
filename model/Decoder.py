import torch
import torch.nn as nn



class Decoder(nn.Module):

    def __init__(self,vocab_size,n_layer=6,d_model=768,nhead=12,max_len=64):
        super().__init__()
        
        self.dropout = nn.Dropout(0.1)
        #text decoder
        self.decoder_embedding = nn.Embedding(vocab_size,d_model)
        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
                d_model,
             nhead,
               batch_first=True
             )
        self.decoder = nn.TransformerDecoder(
             decoder_layer,
             num_layers=n_layer
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(
             d_model,
            vocab_size,
             bias=False
         )

        self.fc.weight = self.decoder_embedding.weight
        #init
        nn.init.normal_(self.decoder_pos_embedding,  std=0.02) 
        
    def forward(self, vlfm_output,target_ids,images,decoder_attention_mask=None,prompt_attention_mask=None):

        #decoder proccess
        target_embedding = self.dropout(
            self.decoder_embedding(target_ids)
            + self.decoder_pos_embedding[:, :target_ids.size(1)]
        )
         #memory mask
        image_mask = torch.zeros(
                     images.size(0),
                    32,
                      dtype=torch.bool,
                     device=images.device
        )  
        #prompt_attention_mask = ~prompt_attention_mask.bool()
        memory_mask = torch.cat(
                    [image_mask, prompt_attention_mask],
                     dim=1
        )
        mask = nn.Transformer.generate_square_subsequent_mask(
            target_ids.size(1)
         ).to(images.device)
        decoder_attention_mask = ~decoder_attention_mask.bool() 
           
        outputs = self.decoder(
           tgt=target_embedding,
             memory=vlfm_output,
             tgt_mask=mask,
             tgt_key_padding_mask=decoder_attention_mask,
              memory_key_padding_mask=memory_mask
        )
           
        outputs = self.norm(outputs)
        outputs = self.fc(outputs)
        return outputs
            