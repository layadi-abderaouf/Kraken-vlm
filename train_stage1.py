import torch
import os
from model.QFormerS1 import QFormerStage1
from tqdm import tqdm
from torch.utils.data import DataLoader,RandomSampler
from dataset import Vision_Dataset
from model.CLIP import CLIP_encoder
import config



def train_stage1(model, dataloader, optimizer, device,scaler, epochs=5):
    """
    تدريب Stage 1 الكامل.

    في كل step:
        loss_total = loss_itc + loss_itm + loss_itg
    """
    model.train()
    clip = CLIP_encoder().to(device)
    best_avg_loss = 100

    for epoch in range(epochs):
        total_loss = 0
        total_itc = 0
        total_itm = 0
        total_itg = 0

        progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{epochs}",
                total=len(dataloader)
            )

        for images, prompt_ids, prompt_mask, target_ids, target_mask, *_ in progress_bar:
            images = images.to(device, non_blocking=True)
            captions_ids = target_ids.to(device, non_blocking=True)
            captions_mask = target_mask.to(device, non_blocking=True)
            with torch.no_grad():
               images = clip(images)

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
               # Forward: الـ 3 objectives معاً
               losses = model(images, captions_ids, captions_mask, mode='all')
               loss = losses['loss_total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_itc += losses['loss_itc'].item()
            total_itm += losses['loss_itm'].item()
            total_itg += losses['loss_itg'].item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}",
                                     itm_loss=f"{losses['loss_itm'].item():.4f}",
                                     itc_loss=f"{losses['loss_itc'].item():.4f}",
                                     itg_loss=f"{losses['loss_itg'].item():.4f}")

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_avg_loss:
            print("save model")
            best_avg_loss = avg_loss
            torch.save(model.state_dict(), "QformerS1.pth")

        n = len(dataloader)
        print(f"\n{'='*50}")
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Avg Total Loss: {total_loss/n:.4f}")
        print(f"  Avg ITC Loss:   {total_itc/n:.4f}")
        print(f"  Avg ITM Loss:   {total_itm/n:.4f}")
        print(f"  Avg ITG Loss:   {total_itg/n:.4f}")
        print(f"{'='*50}")


# ============================================================
# 4. التشغيل
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # إنشاء النموذج
    model = QFormerStage1(
        vocab_size=30522,
        vision_dim=768,         
        d_model=768,
        num_queries=32,
        num_layers=8,
        n_heads=12,
        dropout=0.1,
        max_text_len=64,
        cross_attention_frequency=2,
        pad_token_id=0
    ).to(device)

    if os.path.exists(config.CKPT_QFORMER_S1):
        try:
            model.load_state_dict(torch.load(config.CKPT_QFORMER_S1, map_location=device, weights_only=True))
            print("Loading saved model...")
            
        except RuntimeError:
            print("Saved weights incompatible — training from scratch.")
    else :
        print('from scratch')
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    # Dataset
    dataset = Vision_Dataset(images_folder="/",
                             dataset_file=r"VG\gold\exodus_flickr_caption.jsonl")
    sampler = RandomSampler(
       dataset,
       replacement=False,
       num_samples=250000
    )
    dataloader = DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    #scaler
    scaler = torch.amp.GradScaler("cuda")
    # تدريب
    train_stage1(model, dataloader, optimizer, device, scaler,epochs=10 )
    torch.save(model.qformer.state_dict(), "Qformer.pth")
    
