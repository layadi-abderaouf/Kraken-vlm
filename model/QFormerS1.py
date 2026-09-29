import torch
import torch.nn as nn
import torch.nn.functional as F
from .QFormer import QFormer
from transformers import BertModel
import math





# ============================================================
# 2. Stage 1 Module (منفصل تماماً عن QFormer Core)
# ============================================================

class QFormerStage1(nn.Module):
    """
    وحدة Stage 1 للتدريب.

    تحتوي على:
    - QFormer Core
    - Text Embeddings + Positional Encoding
    - ITC Head  (Image-Text Contrastive)
    - ITM Head  (Image-Text Matching)
    - ITG Head  (Image-Grounded Text Generation)
    - الـ 3 Attention Masks (ITC / ITM / ITG)
    """

    def __init__(
        self,
        vocab_size=30522,
        vision_dim=768,
        d_model=768,
        num_queries=32,
        num_layers=8,
        n_heads=12,
        dropout=0.1,
        max_text_len=77,
        cross_attention_frequency=2,
        pad_token_id=0,
        bert_model_name="google/bert_uncased_L-4_H-512_A-8"
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_text_len = max_text_len

        # === QFormer Core (نقي) ===
        self.qformer = QFormer(
            vision_dim=vision_dim,
            d_model=d_model,
            num_queries=num_queries,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            cross_attention_frequency=cross_attention_frequency
        )

        # === Text Embeddings ===
        bert = BertModel.from_pretrained(bert_model_name)
        self.bert_embeddings = bert.embeddings 
        # Freeze embeddings
        for param in self.bert_embeddings.parameters():
            param.requires_grad = False

        self.bert_embeddings.eval()         
        bert_hidden = bert.config.hidden_size
        del bert

        self.text_proj_in = (
            nn.Linear(bert_hidden, d_model) if bert_hidden != d_model else nn.Identity()
        )

        # === ITC Heads (Contrastive) ===
        self.itc_temp = nn.Parameter(torch.ones([]) * 0.07)  # Learnable temperature
        self.itc_query_proj = nn.Linear(d_model, d_model)
        self.itc_text_proj = nn.Linear(d_model, d_model)

        # === ITM Head (Matching Classifier) ===
        self.itm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # match / no-match
        )

        # === ITG Head (Text Generation) ===
        self.itg_head = nn.Linear(d_model, vocab_size)

        
    

    def get_text_embeddings(self, text_input_ids):
        B, T = text_input_ids.shape
        T = min(T, self.max_text_len)
        embeds = self.bert_embeddings(input_ids=text_input_ids[:, :T])
        return self.text_proj_in(embeds)

    
    


    @torch.no_grad()
    def sample_hard_negatives(self, sim_i2t):

        B = sim_i2t.size(0)

        sim = sim_i2t.clone()

        diagonal = torch.arange(
            B,
            device=sim.device
        )

        sim[diagonal, diagonal] = -float("inf")

        # تحويل similarities إلى probabilities
        weights = F.softmax(
            sim,
            dim=1
        )

        negative_indices = torch.multinomial(
            weights,
            num_samples=1
        ).squeeze(1)

        return negative_indices
    # ----------------------------------------------------------
    # ITC: Image-Text Contrastive
    # ----------------------------------------------------------

    def forward_itc(self, image_features, text_input_ids,text_attention_mask):
        """
        Image-Text Contrastive Learning.

        الهدف: تعليم الـ Queries كيف تستخرج تمثيلات بصرية
        متوافقة مع تمثيلات النص (بدون تداخل).
        """
        text_embeds = self.get_text_embeddings(text_input_ids)
        text_len = text_embeds.size(1)

        # Unimodal mask: Queries و Text منفصلان تماماً
        attn_mask = self.qformer.create_attention_mask(text_len, 'itc', image_features.device)

        B = text_attention_mask.size(0)
        query_padding = torch.zeros(B, self.num_queries, dtype=torch.bool, device=image_features.device)
        text_padding = (text_attention_mask == 0)
        combined_padding = torch.cat([query_padding, text_padding], dim=1)
        # Forward
        query_out, text_out = self.qformer(
            image_features, text_embeds,
            self_attn_mask=attn_mask,
            text_key_padding_mask=combined_padding,
            return_full_output=True
        )

        # Mean pooling
        query_feat = query_out.mean(dim=1)   # [B, D]
        # ========== Text: MASKED mean pooling (الإصلاح!) ==========
        # نوسع الـ mask ليكون [B, T, 1]
        text_mask_expanded = text_attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        # نضرب في الـ mask ونجمع
        text_sum = (text_out * text_mask_expanded).sum(dim=1)  # [B, D]
        # نحسب عدد التوكنات الحقيقية (نتجنب القسمة على صفر)
        text_count = text_attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        text_feat = text_sum / text_count  # [B, D]
        

        # Project & L2 Normalize
        query_feat = F.normalize(self.itc_query_proj(query_feat), dim=-1)
        text_feat = F.normalize(self.itc_text_proj(text_feat), dim=-1)

        # Similarity matrix [B, B]
        sim_i2t = query_feat @ text_feat.T / self.itc_temp
        sim_t2i = sim_i2t.T

        # Labels: diagonal = positive pairs
        B = query_feat.size(0)
        labels = torch.arange(B, device=query_feat.device)

        # InfoNCE loss (bidirectional)
        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)
        loss_itc = (loss_i2t + loss_t2i) / 2

        return loss_itc, sim_i2t

    # ----------------------------------------------------------
    # ITM: Image-Text Matching
    # ----------------------------------------------------------

    def forward_itm(self, image_features, text_input_ids,text_attention_mask ,negative_indices):
        """
        Image-Text Matching.

        الهدف: تعليم Q-Former كيف يميز بين الأزواج المتطابقة
        وغير المتطابقة.

        itm_labels: 1 = match, 0 = no-match
        """
        B = image_features.size(0)
        query_padding = torch.zeros(B * 2, self.num_queries, dtype=torch.bool, device=image_features.device)

        # ==================================================
        # Positive pairs
        # ==================================================
        pos_images = image_features
        pos_text_ids = text_input_ids
        pos_text_mask = text_attention_mask            # [B, T]
        # ==================================================
        # Hard Negative pairs
        # ==================================================
        neg_text_ids = text_input_ids[
            negative_indices
         ]
        neg_text_mask = text_attention_mask[negative_indices] # [B, T]
        # ==================================================
        # Combine Positive + Negative
        # ==================================================
        all_images = torch.cat([ pos_images,pos_images ],dim=0 )
        all_text_ids = torch.cat(
            [
               pos_text_ids,
                neg_text_ids
            ],
            dim=0
        )
        all_text_mask = torch.cat([pos_text_mask, neg_text_mask], dim=0)


        text_embeds = self.get_text_embeddings(all_text_ids)
        text_len = text_embeds.size(1)

        # Bidirectional mask: الكل يرى الكل
        attn_mask = self.qformer.create_attention_mask(text_len, 'itm', image_features.device)

        all_text_padding = (all_text_mask == 0) 
        combined_padding = torch.cat([query_padding, all_text_padding], dim=1)
        # Forward
        query_out = self.qformer(
            all_images, text_embeds,
            self_attn_mask=attn_mask,
            text_key_padding_mask=combined_padding,
            return_full_output=False
        )

        # Pool over ALL tokens (Queries + Text)
        pooled = query_out.mean(dim=1)  # [B, D]

        # ==================================================
        # Labels
        # ==================================================

        labels = torch.cat(
            [
                torch.ones(
                    B,
                    dtype=torch.long,
                    device=image_features.device
                ),
                torch.zeros(
                    B,
                   dtype=torch.long,
                    device=image_features.device
                )
            ]
        )

        # Classify
        logits = self.itm_head(pooled)  # [B, 2]
        loss_itm = F.cross_entropy(logits, labels)

        return loss_itm, logits

    # ----------------------------------------------------------
    # ITG: Image-Grounded Text Generation
    # ----------------------------------------------------------

    def forward_itg(self, image_features, text_input_ids,text_attention_mask):
        """
        Image-Grounded Text Generation.

        الهدف: تعليم Q-Former توليد نص من الصورة.
        الـ Queries تعمل كـ Visual Prefix للـ Decoder.
        """
        text_embeds = self.get_text_embeddings(text_input_ids)
        text_len = text_embeds.size(1)

        # Causal mask: الـ Text يرى Queries + نفسه بشكل Causal
        attn_mask = self.qformer.create_attention_mask(text_len, 'itg', image_features.device)

         # ========== key_padding_mask للـ combined (queries + text) ==========
        B = text_attention_mask.size(0)
        # Queries: لا padding (كلها False = لا ignore)
        query_padding = torch.zeros(B, self.num_queries, dtype=torch.bool, device=image_features.device)
        # Text: padding = True (ignore)
        text_padding = (text_attention_mask == 0)
        combined_key_padding_mask = torch.cat([query_padding, text_padding], dim=1)  # [B, 32+T]

        # Forward
        query_out, text_out = self.qformer(
            image_features, text_embeds,
            self_attn_mask=attn_mask,
            text_key_padding_mask=combined_key_padding_mask,
            return_full_output=True
        )

        # Predict next token
        logits = self.itg_head(text_out)  # [B, T, vocab_size]

        # Shift: position i predicts token[i+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = text_input_ids[:, 1:].contiguous()

        # CrossEntropy (ignore padding)
        loss_itg = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id
        )

        return loss_itg, logits

    # ----------------------------------------------------------
    # Forward Combined
    # ----------------------------------------------------------

    def forward(self, image_features, text_input_ids, text_attention_mask, mode='all'):
        """
        mode:
            'itc'  -> ITC فقط
            'itm'  -> ITM فقط
            'itg'  -> ITG فقط
            'all'  -> الكل معاً (Stage 1 الكامل)
        """
        if mode == 'itc':
            return self.forward_itc(image_features, text_input_ids,text_attention_mask)
        elif mode == 'itm':
            _, sim_i2t = self.forward_itc(
                image_features,
                text_input_ids,
                text_attention_mask
            )

            negative_indices = self.sample_hard_negatives(
                sim_i2t.detach()
            )
            return self.forward_itm(image_features, text_input_ids, text_attention_mask,negative_indices)
        elif mode == 'itg':
            return self.forward_itg(image_features, text_input_ids,text_attention_mask)
        else:
            # Stage 1 الكامل: 3 خسائر معاً
            loss_itc, sim_i2t  = self.forward_itc(image_features, text_input_ids,text_attention_mask)
            negative_indices = self.sample_hard_negatives(
                sim_i2t.detach()
            )
            loss_itm, _ = self.forward_itm(image_features, text_input_ids, text_attention_mask,negative_indices)
            loss_itg, _ = self.forward_itg(image_features, text_input_ids,text_attention_mask)

            return {
                'loss_itc': loss_itc,
                'loss_itm': loss_itm,
                'loss_itg': loss_itg,
                'loss_total': loss_itc + loss_itm + loss_itg
            }

