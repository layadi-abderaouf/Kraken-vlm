import torch
import os
from model.VLFM_S2 import VLFM_S2
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import Vision_Dataset,MultiTaskBatchScheduler
import config



def save_checkpoint(path, model, optimizer, scaler):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)

def reset_task_stats(tasks):
    task_stats = {
        task: {
            "loss": 0.0,
            **{metric: 0.0 for metric in metrics}
        }
        for task, metrics in tasks.items()
    }

    task_batches = {
        task: 0
        for task in tasks
    }

    return task_stats, task_batches


def train_stage1(model, scheduler, optimizer, device,scaler, steps=200000):
    """
      stage 2
    """
    model.train()
    model.vlfm.text_encoder.eval()
    model.vlfm.vision.clip.eval()
    
    
    best_acc = 0 
    progress_bar = tqdm(range(steps), desc="Training")
    
    tasks = {
        "classification": ["acc"],
        "counting": ["acc"],
        "attributes": ["acc"],
        "relationship": ["acc"],
        "caption": ["itm_acc", "mlm_acc"],
    }
    task_stats = {
        task: {
            "loss": 0.0,
            **{metric: 0.0 for metric in metrics}
        }
        for task, metrics in tasks.items()
    }
    task_batches = {
        task: 0
        for task in tasks
    }
    
    for step in progress_bar:
        task, batch = scheduler.get_batch()
        (
            image,
            prompt_input_ids,
            prompt_attention_mask,
            target_input_ids,
            target_attention_mask,
            *_
        ) = batch

        image = image.to(device)
        prompt_input_ids = prompt_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)
        target_input_ids = target_input_ids.to(device)
        target_attention_mask = target_attention_mask.to(device)
        

        optimizer.zero_grad()
            
        with torch.autocast(device_type="cuda", dtype=torch.float16):
             # Forward: الـ 5 objectives معاً
             loss , acc = model(image, prompt_input_ids,
                           target_input_ids,task ,
                           prompt_attention_mask,target_attention_mask)
             

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        task_stats[task]["loss"] += loss.item()
        task_batches[task] += 1

        for metric in tasks[task]:
            value = acc[metric]
            if torch.is_tensor(value):
                value = value.item()
            if value != -1:
               task_stats[task][metric] += acc[metric]
            
        

        progress_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 task=f"{task}") 
        
        if step % 6000 == 0 and step != 0:
            overall_acc = 0
            print(f"\n{'='*50}")
            for task in tasks:
                #compute average loss and acc
                n = task_batches[task]
                if n == 0:
                    continue
                task_stats[task]["loss"] /= n
                for metric in tasks[task]:
                    if task == "caption":
                       task_stats[task][metric] /= (n/2)
                    else:
                       task_stats[task][metric] /= n
                    
                #print loss and acc
                stats = task_stats[task]
                print(f"\n{task}:")
                print(f"  Loss: {stats['loss']:.4f}")
                for metric in tasks[task]:
                   print(f"  {metric}: {stats[metric]:.4f}")
                   overall_acc += stats[metric]
            print(f"{'='*50}")
            overall_acc /= 6
            if overall_acc > best_acc:
               best_acc = overall_acc
               print("save model")
               save_checkpoint(config.CKPT_VLFM_S2,model,optimizer,scaler)
            task_stats, task_batches = reset_task_stats(tasks)
      


# ============================================================
# 4. التشغيل
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # إنشاء النموذج
    model = VLFM_S2().to(device)
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    #scaler
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume logic ----
    if os.path.exists(config.CKPT_VLFM_S2):
            try:
                ckpt = torch.load(config.CKPT_VLFM_S2, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                try:
                   model.vlfm.vision.q_former.load_state_dict(torch.load("Qformer.pth", map_location=device, weights_only=True))
                except:
                    print("qformer failed")
                
            except (RuntimeError, KeyError) as e:
                print(f"Checkpoint incompatible ({e}) — training from scratch.")
    else:    
        print('from scratch')
        
        
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    # Datasets
    classification_dataset = Vision_Dataset(
       images_folder=r"VG",
       dataset_file=r"VG\gold\objects_train_dataset.jsonl",
    )
    classification_dataloader = DataLoader(
        classification_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    counting_dataset = Vision_Dataset(
        images_folder=r"COCO",
        dataset_file=r"VG\gold\counting_train_dataset.jsonl",
    )
    counting_dataloader = DataLoader(
        counting_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    attributes_dataset = Vision_Dataset(
        images_folder=r"VG",
        dataset_file=r"VG\gold\attributes_train_dataset.jsonl",
    )
    attributes_dataloader = DataLoader(
        attributes_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    relationships_dataset = Vision_Dataset(
        images_folder=r"VG",
        dataset_file=r"VG\gold\relationships_train_dataset.jsonl",
    )
    relationships_dataloader = DataLoader(
        relationships_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
       persistent_workers=True
    )

    caption_dataset = Vision_Dataset(
        images_folder=r"D:\Datasets",
        dataset_file=r"VG\gold\caption_train_dataset.jsonl",
    )
    caption_dataloader = DataLoader(    
        caption_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    #Batch shecdular
    loaders = {
        "caption": caption_dataloader,
        "classification": classification_dataloader,
        "relationship": relationships_dataloader,
        "counting": counting_dataloader,
        "attributes": attributes_dataloader,
    }

    weights = {
        "caption": 1,
        "classification": 1,
        "relationship": 1,
        "counting": 1,
        "attributes": 1,
    }

    scheduler = MultiTaskBatchScheduler(
        loaders,
        weights
    )

   
    # تدريب
    train_stage1(model, scheduler, optimizer, device, scaler )
    torch.save(model.vlfm.state_dict(), config.CKPT_VLFM)
    
