import torch
import json
import re
import random
import torch.nn as nn
from transformers import BertModel, BertTokenizer, CLIPVisionModel
from transformers import CLIPImageProcessor
import torch.nn.functional as F
from .VLFM import VLFM

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
    
    
class VLFM_S2(nn.Module):

    def __init__(self, d_model=768, n_attributes=48,n_relationships=5407):
        super().__init__()

        self.vlfm = VLFM()
        
        self.max_text_len = 64
        self.tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")

        # === Classification Head ===
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)
        )

        # === Counting Head ===
        self.counting_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 16)
        )

        # === Attributes Head ===
        self.attribute_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_attributes)
        )

        # === Relationship Head ===
        self.relationship_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_relationships)
        )

        # === ITC Heads ===
        self.itc_temp = nn.Parameter(torch.ones([]) * 0.07)

        self.itc_query_proj = nn.Linear(
            d_model,
            d_model
        )

        self.itc_text_proj = nn.Linear(
            512,
            d_model
        )

        # === ITM Head ===
        self.itm_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)
        )
        # === MLM Head (يعمل على كل token على حدة، بدون pooling) ===
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.tokenizer.vocab_size)
        )

        
        # === POOLING ATTENTION ===
        self.pool_classification = AttentionPooling(d_model)
        self.pool_attributes = AttentionPooling(d_model)
        self.pool_relationship = AttentionPooling(d_model)
        self.pool_counting = AttentionPooling(d_model)
        self.pool_caption = AttentionPooling(d_model)
        
        # === Text Embeddings ===
        bert = BertModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
        self.bert_embeddings = bert.embeddings 
        # Freeze embeddings
        for param in self.bert_embeddings.parameters():
            param.requires_grad = False
        
        self.bert_embeddings.eval()         
        del bert
       
        
        
        with open(
            r"D:\datasets\VG\gold\attributes_index.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.attribute_index = json.load(f)
            
        with open(
                    r"D:\datasets\VG\gold\relationship_index.json",
                    "r",
                    encoding="utf-8"
        ) as f:
                self.relationships_index = json.load(f)
            
        
        # ============================================================
        # Build word -> id mapping
        # ============================================================

        self.attribute_to_id = {
            self.normalize_label(word): int(info["id"])
            for word, info in self.attribute_index.items()
        }
        self.relationships_to_id = {
            self.normalize_label(word): int(info["id"])
            for word, info in self.relationships_index.items()
        }


    # ============================================================
    # GET LABELS
    # ============================================================

    def normalize_label(self,text):
       """
       Normalize target text for label lookup.
       """
       text = str(text).lower().strip()
       # Normalize multiple spaces
       text = re.sub(r"\s+", " ", text)
       # Normalize spaces around hyphens
       text = re.sub(r"\s*-\s*", "-", text)
       return text


    def get_labels(self, target_ids, task):
        """
        Convert tokenized target text into task-specific labels.

        target_ids:
            Tensor [B, T] or [T]

        task:
            classification / counting / attributes / relationships

        Unknown targets:
            -> label = -1
            -> training does NOT crash

        Returns:
            Tensor [B]
        """
        # --------------------------------------------------------
        # Single sample -> batch
        # --------------------------------------------------------
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
        # --------------------------------------------------------
        # Decode
        # --------------------------------------------------------
        decoded = self.tokenizer.batch_decode(
            target_ids,
            skip_special_tokens=True
        )
        labels = []
        # Used only for displaying unknown targets once
        if not hasattr(self, "_unknown_labels"):
            self._unknown_labels = set()
        # ========================================================
        # CLASSIFICATION
        # ========================================================
        if task == "classification":
            for text in decoded:
                text = self.normalize_label(text)
                if text == "yes":
                    labels.append(1)
                elif text == "no":
                    labels.append(0)
                else:
                    if text not in self._unknown_labels:
                        print(
                            f"[WARNING] Unknown classification target: "
                            f"'{text}'"
                        )
                        self._unknown_labels.add(text)
                    labels.append(-1)
        # ========================================================
        # COUNTING
        # ========================================================
        elif task == "counting":
            for text in decoded:
                text = text.strip()
                try:
                    value = int(text)
                    labels.append(value)
                except (ValueError, TypeError):
                    if text not in self._unknown_labels:
                        print(
                            f"[WARNING] Invalid counting target: "
                            f"'{text}'"
                        )
                        self._unknown_labels.add(text)

                    labels.append(-1)
        # ========================================================
        # ATTRIBUTES
        # ========================================================
        elif task == "attributes":
            for text in decoded:
                text = self.normalize_label(text)
                if text in self.attribute_to_id:
                    labels.append(
                        self.attribute_to_id[text]
                    )
                else:
                    if text not in self._unknown_labels:
                        print(
                           f"[WARNING] Unknown attribute target: "
                            f"'{text}'"
                        )
                        self._unknown_labels.add(text)
                    labels.append(-1)
        # ========================================================
        # RELATIONSHIPS
        # ========================================================
        elif task == "relationships":
            for text in decoded:
                text = self.normalize_label(text)
                if text in self.relationships_to_id:
                    labels.append(
                        self.relationships_to_id[text]
                    )
                else:
                    if text not in self._unknown_labels:
                        print(
                            f"[WARNING] Unknown relationship target: "
                            f"'{text}'"
                        )
                        self._unknown_labels.add(text)
                    labels.append(-1)
        # ========================================================
        # INVALID TASK
        # ========================================================
        else:
            raise ValueError(
                f"Unknown task: {task}"
            )
        # ========================================================
        # RETURN
        # ========================================================
        return torch.tensor(
            labels,
            dtype=torch.long,
            device=target_ids.device
        )

    # ============================================================
    # Classification
    # ============================================================

    def classification_forward(self, features,target_id,mask=None):
        # features: [B, N, D]
        pooled = self.pool_classification(features,mask)
        logits = self.classification_head(pooled)
        # logits: [B, 2]
        
        labels = self.get_labels(
                    target_id,
                    task="classification",
                )
        loss = F.cross_entropy(logits, labels)
        
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)

        acc = correct / total

        return loss , acc

    # ============================================================
    # Counting
    # ============================================================

    def counting_forward(self, features,target_id,mask=None):
        # features: [B, N, D]
        pooled = self.pool_counting(features,mask)
        logits = self.counting_head(pooled)
        # logits: [B, 20]
        # classes = 0 ... 19

        labels = self.get_labels(
                target_id,
                task="counting",
        )
        loss = F.cross_entropy(logits, labels)
                
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        
        acc = correct / total
        
        return loss , acc

    # ============================================================
    # Attributes
    # ============================================================

    def attribute_forward(self, features,target_id,mask=None):
        # features: [B, N, D]
        pooled = self.pool_attributes(features,mask)
        logits = self.attribute_head(pooled)
        # logits: [B, n_attributes]

        labels = self.get_labels(
            target_id,
            task="attributes",
        )
        valid_mask = labels >= 0
        if not valid_mask.any():
            return (
                torch.tensor(
                    0.0,
                    device=features.device,
                    requires_grad=True
                ),
                0.0
            )

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        loss = F.cross_entropy(valid_logits, valid_labels)
                
        predictions = logits.argmax(dim=-1)
        valid_predictions = predictions[valid_mask]
        correct = (valid_predictions == valid_labels).sum().item()
        total = valid_labels.size(0)
        
        acc = correct / total
        
        return loss , acc

    # ============================================================
    # Relationships
    # ============================================================

    def relationship_forward(self, features,target_id,mask=None):
        # features: [B, N, D]
        pooled = self.pool_relationship(features,mask)
        logits = self.relationship_head(pooled)
        # logits: [B, vocab_size]
        labels = self.get_labels(
                    target_id,
                    task="relationships",
                )
        valid_mask = labels >= 0
        if not valid_mask.any():
            return (
                torch.tensor(
                    0.0,
                       device=features.device,
                       requires_grad=True
                 ),
                0.0
            )
        
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        loss = F.cross_entropy(valid_logits, valid_labels)
        
        predictions = logits.argmax(dim=-1)
        valid_predictions = predictions[valid_mask]
        correct = (valid_predictions == valid_labels).sum().item()
        total =  valid_labels.size(0)  # ✅ يحسب الـ valid فقط
        acc = correct / total

        return loss , acc
    
    # ============================================================
    # MLM Masking (80/10/10 strategy)
    # ============================================================
    def mask_tokens(self, input_ids, attention_mask, mlm_probability=0.15):
        """
        input_ids      : [B, T]
        attention_mask : [B, T]  (1 = real token, 0 = padding)

        return:
            masked_input_ids : [B, T]  (النسخة المُدخلة للنموذج)
            labels           : [B, T]  (-100 في المواقع غير المخفية، تُتجاهل في الخسارة)
        """
        labels = input_ids.clone()

        # احتمال الإخفاء لكل موضع
        probability_matrix = torch.full(
            labels.shape, mlm_probability, device=input_ids.device
        )

        # لا نخفي التوكنز الخاصة ([CLS], [SEP], [PAD], ...)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True
            )
            for val in input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool, device=input_ids.device
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # لا نخفي الـ padding
        if attention_mask is not None:
            probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        # تحديد المواقع المخفية فعلياً (15%)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # الخسارة تُحسب فقط على المواقع المخفية
        labels[~masked_indices] = -100

        masked_input_ids = input_ids.clone()

        # 80% من المخفية -> [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool()
            & masked_indices
        )
        masked_input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% من المخفية -> توكن عشوائي (النصف المتبقي بعد الـ80%)
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape,
            dtype=torch.long, device=input_ids.device
        )
        masked_input_ids[indices_random] = random_words[indices_random]

        # الـ 10% الأخيرة تبقى بدون تغيير (لا حاجة لكود إضافي هنا)

        return masked_input_ids, labels
    
    # ============================================================
    # MLM Forward
    # ============================================================
    def mlm_forward(self, images, prompt_ids,prompt_attention_mask,
                    caption_ids, caption_attention_mask):
        
       
        # 1) إخفاء عشوائي على الكابشن
        masked_caption_ids, caption_labels = self.mask_tokens(
           caption_ids,
           caption_attention_mask
        )
        
        all_text_ids = torch.cat(
            [ prompt_ids,masked_caption_ids ],dim=1)
        all_text_mask = torch.cat(
            [prompt_attention_mask,caption_attention_mask  ],  dim=1)

        # 2) تمرير الصورة + النص المُخفى عبر VLFM
        fusion_output = self.vlfm(images, all_text_ids, all_text_mask)
        # fusion_output: [B, N, D]  حيث N = image_tokens + T

        # 3) استخراج تمثيلات النص فقط (بعد tokens الصورة)
        T = all_text_ids.size(1)
        num_image_tokens = fusion_output.size(1) - T
        text_features = fusion_output[:, num_image_tokens:, :]  # [B, T, D]

        # 4) التنبؤ بكل توكن
        logits = self.mlm_head(text_features)  # [B, T, vocab_size]
        
        #labels
        prompt_labels = torch.full_like(
           prompt_ids,
           -100
        )
        mlm_labels = torch.cat(
           [prompt_labels, caption_labels],
           dim=1
        )

        # 5) الخسارة فقط على المواقع المخفية (-100 تُتجاهل تلقائياً)
        loss_mlm = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        # 6) الدقة على المواقع المخفية فقط
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = mlm_labels != -100
            if mask.sum() > 0:
                acc = (preds[mask] == mlm_labels[mask]).float().mean().item()
            else:
                acc = 0.0

        return loss_mlm, acc

    # ============================================================
    # ITM
    # ============================================================

    def itm_forward(self,images,
        prompt_ids,
        prompt_attention_mask,
        caption_ids,
        caption_attention_mask,
        negative_indices):

        # features: [B, N, D]
        B = images.size(0)
        # ============================================================
        # Positive
        # ============================================================
        pos_images = images
        pos_prompt_ids = prompt_ids
        pos_prompt_mask = prompt_attention_mask
        pos_caption_ids = caption_ids
        pos_caption_mask = caption_attention_mask
        # ============================================================
        # Hard Negative Caption
        # ============================================================
        neg_images = images
        neg_prompt_ids = prompt_ids
        neg_prompt_mask = prompt_attention_mask
        neg_caption_ids = caption_ids[
            negative_indices
        ]
        neg_caption_mask = caption_attention_mask[
            negative_indices
        ]
        # ============================================================
        # Combine
        # ============================================================
        all_images = torch.cat(
            [pos_images,neg_images ],  dim=0)
        all_prompt_ids = torch.cat(
            [ pos_prompt_ids, neg_prompt_ids ], dim=0)
        all_prompt_mask = torch.cat(
            [ pos_prompt_mask,  neg_prompt_mask ],  dim=0)
        all_caption_ids = torch.cat(
            [ pos_caption_ids,neg_caption_ids], dim=0)
        all_caption_mask = torch.cat(
            [ pos_caption_mask,neg_caption_mask ],dim=0 )
        # ============================================================
        # Instruction + Caption
        # ============================================================
        all_text_ids = torch.cat(
            [ all_prompt_ids,all_caption_ids ],dim=1)
        all_text_mask = torch.cat(
            [all_prompt_mask,all_caption_mask  ],  dim=1)
        # ============================================================
        # VLFM
        # ============================================================
        fusion_output = self.vlfm(
            all_images,
            all_text_ids,
            all_text_mask
        )
        # [2B, N, D]
        # ============================================================
        # Pool
        # ============================================================
        pooled = self.pool_caption(fusion_output)
        # [2B, D]

        # ============================================================
        # ITM Classification
        # ============================================================
        logits = self.itm_head(
            pooled
        )
        # [2B, 2]

        # ============================================================
        # Labels
        # ============================================================
        labels = torch.cat(
            [
                torch.ones(
                    B,
                    dtype=torch.long,
                    device=images.device
                ),
                torch.zeros(
                    B,
                    dtype=torch.long,
                    device=images.device
                )
            ],
            dim=0
        )
        # ============================================================
        # Loss
        # ============================================================
        loss_itm = F.cross_entropy(
            logits,
            labels
        )
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        acc = correct / total

        return loss_itm , acc

    # ============================================================
    # ITC
    # ============================================================
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
        
    def get_text_embeddings(self, text_input_ids):
            B, T = text_input_ids.shape
            T = min(T, self.max_text_len)
            embeds = self.bert_embeddings(input_ids=text_input_ids[:, :T])
            return self.itc_text_proj(embeds)

    def itc_forward(self,
        vlfm_output,
        caption_ids,
        caption_attention_mask,mask=None):

        # vlfm_output: [B, N, D]

        # Pool image tokens
        vlfm_output = self.pool_caption(vlfm_output,mask)
        vlfm_output = self.itc_query_proj(
                    vlfm_output
                )

        # Pool text tokens
        text_embeds = self.get_text_embeddings(
            caption_ids
        )
        text_features = self.pool_caption(text_embeds,caption_attention_mask)

        # Normalize
        vlfm_output = F.normalize(
            vlfm_output,
            dim=-1
        )
        text_features = F.normalize(
            text_features,
            dim=-1
        )

        # Similarity 
        temperature = self.itc_temp.clamp(
            min=0.001,
            max=0.5
        )
        sim_i2t = (
            vlfm_output @ text_features.T
        ) / temperature
        sim_t2i = sim_i2t.T
        
        # Labels: diagonal = positive pairs
        B = vlfm_output.size(0)
        labels = torch.arange(B, device=vlfm_output.device)
        #loss
        loss_i2t = F.cross_entropy( sim_i2t , labels)
        loss_t2i = F.cross_entropy( sim_t2i, labels )
        loss_itc = (
            loss_i2t + loss_t2i
        ) / 2

        
        return loss_itc , sim_i2t ,sim_t2i

    def build_fusion_mask(self, images, prompt_attention_mask):
       """
       يبني attention mask يطابق طول vlfm_output
       (image tokens + text tokens)
       """
       B = images.size(0)
       # عدد tokens الصورة (ثابت عادة، كل الصورة valid)
       num_image_tokens = 32  # عدّل الاسم حسب تعريفك الفعلي في VLFM

       image_mask = torch.ones(
           B, num_image_tokens,
           dtype=prompt_attention_mask.dtype,
           device=prompt_attention_mask.device
       )

       fusion_mask = torch.cat([image_mask, prompt_attention_mask], dim=1)
       return fusion_mask
   
   
    def forward(self,
            images,
            prompt_ids,
            target_ids,
            task,
            prompt_attention_mask=None,
            target_attention_mask=None,
            
    ):
        vlfm_output = self.vlfm(
                    images,
                    prompt_ids,
                    prompt_attention_mask
                )
        fusion_mask = None
        if prompt_attention_mask is not None:
            fusion_mask = self.build_fusion_mask(images, prompt_attention_mask)
        # ============================================================
        # caption
        # ============================================================
        if task == "caption":
            selected_task = random.randint(0, 1)
            if selected_task == 0:
                itc_loss , sim_i2t , _ = self.itc_forward(
                    vlfm_output=vlfm_output,
                    caption_ids=target_ids,
                    caption_attention_mask=target_attention_mask,
                    mask=fusion_mask
                )
                negative_indices = (
                        self.sample_hard_negatives(sim_i2t)
                    )
                itm_loss , itm_acc = self.itm_forward(
                    images=images,
                    prompt_ids=prompt_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    caption_ids=target_ids,
                    caption_attention_mask=target_attention_mask,
                    negative_indices=negative_indices
                )
                return  itm_loss+itc_loss ,{"itm_acc": itm_acc, "mlm_acc": -1}
            else:
                mlm_loss, mlm_acc = self.mlm_forward(
                    images=images,
                    prompt_ids=prompt_ids,
                    prompt_attention_mask=prompt_attention_mask,
                   caption_ids=target_ids,
                    caption_attention_mask=target_attention_mask
                )
                return  mlm_loss ,{"itm_acc": -1, "mlm_acc": mlm_acc}

        # ============================================================
        # classification
        # ============================================================
        if task == "classification":
            loss , acc = self.classification_forward(vlfm_output,target_ids,mask=fusion_mask)
            return loss , {"acc": acc}
        # ============================================================
        # classification
        # ============================================================
        if task == "counting":
            loss , acc = self.counting_forward(vlfm_output,target_ids,mask=fusion_mask)
            return loss ,  {"acc": acc}
        # ============================================================
        # classification
        # ============================================================
        if task == "attributes":
            loss , acc = self.attribute_forward(vlfm_output,target_ids,mask=fusion_mask)
            return loss ,  {"acc": acc}
        # ============================================================
        # classification
        # ============================================================
        if task == "relationship":
            loss , acc = self.relationship_forward(vlfm_output,target_ids,mask=fusion_mask)
            return loss ,  {"acc": acc}
                

        