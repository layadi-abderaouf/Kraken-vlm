import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import os

class AttentionPooling(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, features,attention_mask=None):
        # features: [B, N, D]
        scores = self.score(features)  # [B, N, 1]
        if attention_mask is not None :
            mask = attention_mask.bool().unsqueeze(-1)
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min
            )

        
        weights = torch.softmax(scores, dim=1)

        pooled = (features * weights).sum(dim=1)

        return pooled
    
class Router(nn.Module):

    def __init__(self,d_model=768,n_decoders=6):
        super().__init__()
        
        self.pool = AttentionPooling(d_model=d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        self.head = nn.Linear(
            512,
            n_decoders
        )

    def forward(self, vlfm_output):
        x = self.pool(vlfm_output)
        x = self.mlp(x)
        logits = self.head(x)

        return logits
    
    
    
    
    
    



class MoD(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_layer=6,
        d_model=768,
        nhead=12,
        max_len=64,
        n_decoders=6
    ):
        super().__init__()

        self.d_model = d_model
        self.n_decoders = n_decoders
        self.vocab_size = vocab_size

        # ============================================================
        # Router
        # ============================================================

        self.router = Router(
            d_model=d_model,
            n_decoders=n_decoders
        )

        # ============================================================
        # Decoder embedding
        # ============================================================

        self.dropout = nn.Dropout(0.1)

        self.decoder_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.decoder_pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model)
        )

        # ============================================================
        # Base decoder
        # ============================================================

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layer
        )

        # ============================================================
        # Independent decoders
        # ============================================================

        self.decoders = nn.ModuleList([
            copy.deepcopy(decoder)
            for _ in range(n_decoders)
        ])

        # ============================================================
        # Output
        # ============================================================

        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.fc.weight = self.decoder_embedding.weight

        # ============================================================
        # Load router checkpoint
        # ============================================================

        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "router.pth"
        )

        if os.path.exists(checkpoint_path):

            try:

                self.router.load_state_dict(
                    torch.load(
                        checkpoint_path,
                        map_location="cuda",
                        weights_only=True
                    )
                )

                print("Router: Loading saved model...")

            except RuntimeError:

                print(
                    "Router: Saved weights incompatible — "
                    "training from scratch."
                )

        else:

            print("Router: from scratch")

        # ============================================================
        # Freeze router
        # ============================================================

        for p in self.router.parameters():
            p.requires_grad = False

        self.router.eval()

    # =================================================================
    # Causal mask
    # =================================================================

    def generate_causal_mask(
        self,
        seq_len,
        device
    ):

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        return mask

    # =================================================================
    # Prepare decoder inputs
    # =================================================================

    def _prepare_decoder_inputs(
        self,
        target_ids,
        images,
        prompt_attention_mask
    ):

        # -------------------------------------------------------------
        # Target embedding
        # -------------------------------------------------------------

        target_embedding = self.dropout(
            self.decoder_embedding(target_ids)
            +
            self.decoder_pos_embedding[
                :,
                :target_ids.size(1)
            ]
        )

        # -------------------------------------------------------------
        # Causal mask
        # -------------------------------------------------------------

        causal_mask = self.generate_causal_mask(
            target_ids.size(1),
            target_ids.device
        )

        # -------------------------------------------------------------
        # Image mask
        #
        # VLFM image tokens = 32
        # -------------------------------------------------------------

        image_mask = torch.zeros(
            images.size(0),
            32,
            dtype=torch.bool,
            device=target_ids.device
        )

        # -------------------------------------------------------------
        # Memory mask
        # -------------------------------------------------------------
       
        memory_mask = torch.cat(
            [
                image_mask,
                prompt_attention_mask
            ],
            dim=1
        )

        return (
            target_embedding,
            causal_mask,
            memory_mask
        )

    # =================================================================
    # ORIGINAL FORWARD
    #
    # This remains the normal MoD forward.
    # Router chooses one decoder per sample.
    # =================================================================

    def forward(
        self,
        vlfm_output,
        target_ids,
        images,
        target_attention_mask,
        prompt_attention_mask,
        return_router_info=False,
        task_id=None
    ):

        B = vlfm_output.size(0)

        # -------------------------------------------------------------
        # Router
        # -------------------------------------------------------------
        target_attention_mask[:, 0] = False
        router_logits = self.router(vlfm_output)

        decoder_idx = router_logits.argmax(
            dim=-1
        )

        # -------------------------------------------------------------
        # Prepare inputs
        # -------------------------------------------------------------

        (
            target_embedding,
            causal_mask,
            memory_mask
        ) = self._prepare_decoder_inputs(
            target_ids,
            images,
            prompt_attention_mask
        )

        # -------------------------------------------------------------
        # Output tensor
        # -------------------------------------------------------------

        outputs = torch.empty(
            B,
            target_ids.size(1),
            self.d_model,
            device=target_ids.device,
            dtype=target_embedding.dtype
        )

        # -------------------------------------------------------------
        # Dispatch samples
        # -------------------------------------------------------------

        for decoder_id in range(self.n_decoders):

            indices = torch.where(
                decoder_idx == decoder_id
            )[0]

            if indices.numel() == 0:
                continue

            decoder_input = target_embedding[
                indices
            ]

            memory = vlfm_output[
                indices
            ]

            curr_target_mask = (
                target_attention_mask[indices]
                if target_attention_mask is not None
                else None
            )

            curr_memory_mask = memory_mask[
                indices
            ]

            decoder_output = self.decoders[
                decoder_id
            ](
                tgt=decoder_input,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=curr_target_mask,
                memory_key_padding_mask=curr_memory_mask
            )

            decoder_output = decoder_output.to(
                outputs.dtype
            )

            outputs[
                indices
            ] = decoder_output

        # -------------------------------------------------------------
        # Output projection
        # -------------------------------------------------------------

        outputs = self.norm(outputs)

        outputs = self.fc(outputs)

        # -------------------------------------------------------------
        # Optional router information
        # -------------------------------------------------------------

        if return_router_info:

            return {
                "logits": outputs,
                "router_logits": router_logits,
                "decoder_idx": decoder_idx
            }

        return outputs

     # =================================================================
    # RUN ALL DECODERS
    #
    # Returns:
    #
    # [B, D, S, V]
    #
    # B = batch
    # D = number of decoders
    # S = sequence length
    # V = vocabulary
    # =================================================================

    def forward_all_decoders(
        self,
        vlfm_output,
        target_ids,
        images,
        target_attention_mask,
        prompt_attention_mask
    ):
      
        target_attention_mask[:, 0] = False
      
        (
            target_embedding,
            causal_mask,
            memory_mask
        ) = self._prepare_decoder_inputs(
            target_ids,
            images,
            prompt_attention_mask
        )
      

      

        all_logits = []

        # =============================================================
        # Run every decoder on the ENTIRE batch
        # =============================================================

        for decoder_id in range(self.n_decoders):

            decoder_output = self.decoders[
                decoder_id
            ](
                tgt=target_embedding,
                memory=vlfm_output,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=target_attention_mask,
                memory_key_padding_mask=memory_mask
            )
         

            decoder_output = self.norm(
                decoder_output
            )
          

            logits = self.fc(
                decoder_output
            )
          

            all_logits.append(logits)

        # [D, B, S, V] -> [B, D, S, V]

        all_logits = torch.stack(
            all_logits,
            dim=0
        )

        all_logits = all_logits.transpose(
            0,
            1
        )

        return all_logits

    # =================================================================
    # PER-SAMPLE LOSS FOR EVERY DECODER
    #
    # Returns:
    #
    # [B, D]
    #
    # =================================================================

    def compute_decoder_losses(
        self,
        all_logits,
        labels,
        pad_token_id,
        sep_token_id=None,
        cls_token_id=None
    ):

        B, D, S, V = all_logits.shape
    
       
        # =============================================================
        # Valid tokens
        # =============================================================

        valid = (
            labels != pad_token_id
        )

        if sep_token_id is not None:
            valid &= (
                labels != sep_token_id
            )

        if cls_token_id is not None:
            valid &= (
                labels != cls_token_id
            )

        valid_count = valid.sum(dim=1)

       
        # =============================================================
        # Loss
        # =============================================================

        decoder_losses = []

        for decoder_id in range(D):

            logits = all_logits[:, decoder_id]

            token_loss = F.cross_entropy(
                logits.reshape(-1, V),
                labels.reshape(-1),
                reduction="none"
            )

            token_loss = token_loss.reshape(B, S)
    
            sample_loss = (
                token_loss * valid.float()
            ).sum(dim=1) / valid_count.clamp(min=1).float()

            decoder_losses.append(sample_loss)

        decoder_losses = torch.stack(
            decoder_losses,
            dim=1
        )


      

        return decoder_losses

    # =================================================================
    # CORRECTNESS FOR EVERY DECODER
    #
    # Sequence-level exact match.
    #
    # Returns:
    #
    # [B, D]
    #
    # =================================================================

    def compute_decoder_correctness(
        self,
        all_logits,
        labels,
        pad_token_id,
        sep_token_id=None,
        cls_token_id=None
    ):
        # all_logits: [B, D, S, V]
        # labels:     [B, S]

        predictions = all_logits.argmax(dim=-1)
        # [B, D, S]

        valid = (
            labels != pad_token_id
        )

        if sep_token_id is not None:
            valid &= (
                labels != sep_token_id
            )

        if cls_token_id is not None:
            valid &= (
                labels != cls_token_id
            )

        # [B, D, S]
        correct_tokens = (
            predictions == labels.unsqueeze(1)
        )

        correct_tokens &= valid.unsqueeze(1)

        # Number of correct tokens for each sample/decoder
        # [B, D]
        correct_count = correct_tokens.sum(dim=-1)

        # Number of valid tokens for each sample
        # [B]
        valid_count = valid.sum(dim=-1)

        # Token accuracy for each sample/decoder
        # [B, D]
        decoder_token_acc = (
            correct_count.float()
            /
            valid_count.unsqueeze(1).clamp(min=1).float()
        )

        return decoder_token_acc

    # =================================================================
    # ROUTER ANALYSIS
    #
    # This function does NOT train anything.
    #
    # It only measures:
    #
    # 1. Router vs minimum-loss decoder
    # 2. Router correctness
    # 3. Whether another decoder was correct
    # 4. Loss oracle correctness
    #
    # =================================================================
    @torch.no_grad()
    def analyze_router(
    self,
    vlfm_output,
    target_ids,
    images,
    target_attention_mask,
    prompt_attention_mask,
    labels,
    pad_token_id,
    sep_token_id=None,
    cls_token_id=None
    ):

        # =============================================================
        # Router prediction
        # =============================================================
        
        router_logits = self.router(
            vlfm_output
        )
        
        router_idx = router_logits.argmax(
            dim=-1
        )
        
        # =============================================================
        # Run all decoders
        # =============================================================
        
        all_logits = self.forward_all_decoders(
            vlfm_output=vlfm_output,
            target_ids=target_ids,
            images=images,
            target_attention_mask=target_attention_mask,
            prompt_attention_mask=prompt_attention_mask
        )
        
        # =============================================================
        # Loss
        # =============================================================
        
        decoder_losses = self.compute_decoder_losses(
            all_logits=all_logits,
            labels=labels,
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id
        )
        
        # =============================================================
        # Token Accuracy
        # =============================================================
        
        decoder_token_acc = self.compute_decoder_correctness(
            all_logits=all_logits,
            labels=labels,
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id
        )
        
        # =============================================================
        # Loss Oracle
        # =============================================================
        
        loss_oracle_idx = decoder_losses.argmin(
            dim=-1
        )
        
        # =============================================================
        # Router-selected decoder
        # =============================================================
        
        batch_idx = torch.arange(
            router_idx.size(0),
            device=router_idx.device
        )
        
        router_loss = decoder_losses[
            batch_idx,
            router_idx
        ]
        
        router_token_acc = decoder_token_acc[
            batch_idx,
            router_idx
        ]
        
        # =============================================================
        # Oracle loss
        # =============================================================
        
        oracle_loss = decoder_losses[
            batch_idx,
            loss_oracle_idx
        ]
        
        # =============================================================
        # Loss-oracle Token Accuracy
        # =============================================================
        
        loss_oracle_token_acc = decoder_token_acc[
            batch_idx,
            loss_oracle_idx
        ]
        
        # =============================================================
        # Best Token-Accuracy decoder
        # =============================================================
        
        best_token_acc = decoder_token_acc.max(
            dim=-1
        ).values
        
        is_best_decoder = (
            decoder_token_acc
            ==
            best_token_acc.unsqueeze(1)
        )
        valid_routing_samples = best_token_acc > 0
        # Token accuracy achieved by the best decoder(s)
        token_acc_oracle = best_token_acc
        
        # =============================================================
        # Router == Loss Oracle
        # =============================================================
        
        router_matches_loss_oracle = (
            router_idx
            ==
            loss_oracle_idx
        )
        
        # =============================================================
        # Router == Token Accuracy Oracle
        # =============================================================
        
        router_matches_token_acc_oracle = (
            is_best_decoder[
                batch_idx,
                router_idx
            ]
            & valid_routing_samples
        )
        
        # =============================================================
        # Return
        # =============================================================
        
        return {
        
            "router_logits":
                router_logits,
        
            "router_idx":
                router_idx,
        
            "all_logits":
                all_logits,
        
            "decoder_losses":
                decoder_losses,
        
            "decoder_token_acc":
                decoder_token_acc,
        
            "loss_oracle_idx":
                loss_oracle_idx,
        
            "token_acc_oracle_idx":
                token_acc_oracle,
        
            "router_loss":
                router_loss,
        
            "oracle_loss":
                oracle_loss,
        
            "router_token_acc":
                router_token_acc,
        
            "loss_oracle_token_acc":
                loss_oracle_token_acc,
        
            "token_acc_oracle":
                token_acc_oracle,
        
            "router_matches_loss_oracle":
                router_matches_loss_oracle,
        
            "router_matches_token_acc_oracle":
                router_matches_token_acc_oracle,
            "valid_routing_sample": valid_routing_samples,
        }