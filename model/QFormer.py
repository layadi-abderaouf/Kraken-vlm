import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. QFormer Core (نقي، لا يعرف شيئاً عن Stage 1)
# ============================================================

class QFormerLayer(nn.Module):
    """
    طبقة واحدة من Q-Former.

    التدفق:
        1. Self-Attention (مشترك بين Queries + Text)
        2. Cross-Attention (Queries ← Image) [كل N طبقات]
        3. Feed-Forward Network
    """

    def __init__(self, d_model=768, n_heads=12, dropout=0.1, cross_attention=True):
        super().__init__()
        self.cross_attention_enabled = cross_attention

        # 1. Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross-Attention
        if cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        queries,
        image_features=None,
        self_attn_mask=None,
        self_key_padding_mask=None,
        image_key_padding_mask=None,
        num_queries=32
    ):
        # ---- 1. Self-Attention ----
        residual = queries
        x, _ = self.self_attn(
            query=queries, key=queries, value=queries,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            need_weights=False
        )
        queries = self.norm1(residual + self.dropout1(x))

        # ---- 2. Cross-Attention (للـ Queries فقط) ----
        if self.cross_attention_enabled and image_features is not None:
            query_states = queries[:, :num_queries]
            residual = query_states
            x, _ = self.cross_attn(
                query=query_states, key=image_features, value=image_features,
                key_padding_mask=image_key_padding_mask,
                need_weights=False
            )
            query_states = self.norm2(residual + self.dropout2(x))
            queries = torch.cat([query_states, queries[:, num_queries:]], dim=1)

        # ---- 3. FFN ----
        residual = queries
        x = self.ffn(queries)
        queries = self.norm3(residual + x)

        return queries


class QFormer(nn.Module):
    """
    Q-Former Core.

    المدخلات:
        image_features: [B, N_image, vision_dim]
        text_tokens:    [B, T, d_model] (اختياري)
        self_attn_mask: [L, L] حيث L = num_queries + T

    المخرجات:
        - return_full_output=False: [B, num_queries, d_model]
        - return_full_output=True:  ([B, num_queries, D], [B, T, D])
    """

    def __init__(
        self,
        vision_dim=768,
        d_model=768,
        num_queries=32,
        num_layers=8,
        n_heads=12,
        dropout=0.1,
        cross_attention_frequency=2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable Query Tokens (الأهم!)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

        # Vision projection (لو الأبعاد مختلفة)
        self.vision_projection = (
            nn.Linear(vision_dim, d_model) if vision_dim != d_model else nn.Identity()
        )

        # الطبقات
        self.layers = nn.ModuleList([
            QFormerLayer(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                cross_attention=(i % cross_attention_frequency == 0)
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        
    def create_attention_mask(self, text_len, mode, device):
            """
            إنشاء Attention Mask حسب المهمة.
    
            total_len = num_queries + text_len
            mask shape: [total_len, total_len]
    
            القيم:
                0     = مسموح (allow)
                -inf  = محظور (block)
            """
            total_len = self.num_queries + text_len
            mask = torch.zeros(total_len, total_len, device=device)
    
            if mode == 'itc':
                # ===== Unimodal Mask =====
                # Queries ترى Queries فقط
                # Text  يرى Text فقط
                # لا تفاعل متبادل بينهما
                mask[:self.num_queries, :self.num_queries] = 1
                mask[self.num_queries:, self.num_queries:] = 1
    
            elif mode in ('itm', 'instruct'):
                # ===== Bidirectional Mask =====
                # الكل يرى الكل (Queries ↔ Text)
                mask.fill_(1)
    
            elif mode == 'itg':
                # ===== Causal Mask =====
                # Queries ترى Queries فقط (self-attention)
                mask[:self.num_queries, :self.num_queries] = 1
                # Text يرى Queries (كـ Visual Prefix)
                mask[self.num_queries:, :self.num_queries] = 1
                # Text يرى نفسه بشكل Causal (triangular)
                causal = torch.tril(torch.ones(text_len, text_len, device=device))
                mask[self.num_queries:, self.num_queries:] = causal
    
            # Convert: 1 → 0 (allow), 0 → -inf (block)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
            return mask

    def forward(
        self,
        image_features,
        text_tokens=None,
        self_attn_mask=None,
        image_key_padding_mask=None,
        text_key_padding_mask=None,
        return_full_output=False
    ):
        B = image_features.size(0)

        # Project image features
        image_features = self.vision_projection(image_features)

        # Expand learnable queries
        queries = self.query_tokens.expand(B, -1, -1)

        # دمج النص مع الـ Queries
        if text_tokens is not None:
            combined = torch.cat([queries, text_tokens], dim=1)
        else:
            combined = queries

        # Pass through layers
        for layer in self.layers:
            combined = layer(
                queries=combined,
                image_features=image_features,
                self_attn_mask=self_attn_mask,
                self_key_padding_mask=text_key_padding_mask,
                image_key_padding_mask=image_key_padding_mask,
                num_queries=self.num_queries
            )

        # Extract outputs
        query_out = self.final_norm(combined[:, :self.num_queries, :])

        if return_full_output and text_tokens is not None:
            text_out = self.final_norm(combined[:, self.num_queries:, :])
            return query_out, text_out

        return query_out
    