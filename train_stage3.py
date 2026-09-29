import torch
import torch.nn as nn
import os
import time
from model.model import Kraken_VLM
import bitsandbytes as bnb
from transformers import BertTokenizer
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




def train_stage3_MoD(model, scheduler, optimizer, criterion,device,scaler, mode="decoders",steps=200000):
    """
      stage 3 : phase 1 and 2
    """
    model.train()
    model.vlfm.text_encoder.eval()
    model.vlfm.vision.clip.eval()
    
    best_acc = 0
    progress_bar = tqdm(range(steps), desc="Training")
    
    DECODER_MAP = {
        "classification": 0,
        "counting":       1,
        "attributes":     2,
        "relationship":   3,
        "caption":        4,
        "vqa":            5,

    }   
    
    tasks = {
        "classification": ["acc"],
        "counting": ["acc"],
        "attributes": ["acc"],
        "relationship": ["acc"],
        "vqa": ["acc"],
        "caption": ["acc"],
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
            target_input = target_input_ids[:, :-1]
            target_mask = target_attention_mask[:, :-1]
             # Forward: الـ 5 objectives معاً
            logits = model(image, prompt_input_ids,
                           target_input,
                           prompt_attention_mask,target_mask)
            
            if mode == "decoders":
                labels = target_input_ids[:, 1:]
                loss =  criterion(
                                logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))
                
                predictions = logits.argmax(dim=-1)
                valid = (
                    (labels != tokenizer.pad_token_id) &
                    (labels != tokenizer.sep_token_id) &
                    (labels != tokenizer.cls_token_id)
                )
                correct = ((predictions == labels) & valid).sum().item()
                total = valid.sum().item()
            
                acc = correct / total
                
            elif mode == "router":
                label = DECODER_MAP[task]
                labels = torch.tensor(
                    [label] * logits.size(0),
                    device=logits.device,
                    dtype=torch.long
                )
                loss = criterion(
                    logits,
                    labels
                )
                predictions = logits.argmax(dim=-1)
                correct = (predictions == labels).sum().item()
                total = labels.size(0)
                            
                acc = correct / total
             
        
        
        scaler.scale(loss).backward()
        
       
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        
       

        task_stats[task]["loss"] += loss.item()
        task_batches[task] += 1

        for metric in tasks[task]:
            if acc != -1:
               task_stats[task][metric] += acc
            
        

        progress_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 task=f"{task}") 
        
        if step % 2000 == 0 and step != 0:
            overall_acc = 0
            print(f"\n{'='*50}")
            for task in tasks:
                #compute average loss and acc
                n = task_batches[task]
                if n == 0:
                    continue
                task_stats[task]["loss"] /= n
                for metric in tasks[task]:
                    #if task != "caption":
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
               print("save best model")
               if mode == "router":
                   save_checkpoint(config.CKPT_ROUTER,model,optimizer,scaler)
               else:
                   save_checkpoint(config.CKPT_MOD,model,optimizer,scaler)
            else:
                print("save  model")
                if mode == "router":
                    save_checkpoint(config.CKPT_ROUTER,model,optimizer,scaler)
                else:
                    save_checkpoint(config.CKPT_MOD,model,optimizer,scaler)
            task_stats, task_batches = reset_task_stats(tasks)
            
      


# ============================================================
# 4. التشغيل
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = BertTokenizer.from_pretrained(
                     config.BERT_MODEL
                 )
    # إنشاء النموذج
    model = Kraken_VLM(vocab_size=tokenizer.vocab_size).to(device)
    # Optimizer
    
   
    optimizer = bnb.optim.AdamW8bit(
     [
       {
             "params": [
                 p for p in model.vlfm.parameters()
                 if p.requires_grad
             ],
             "lr": 1e-5
         },
         {
             "params": [
                 p for p in model.decoder.parameters()
                 if p.requires_grad
             ],
             "lr": 1e-4
         }
     ],
    weight_decay=0.05
     )
    criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id
        )
    router_criterion = nn.CrossEntropyLoss()
    #scaler
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume logic ----
    if os.path.exists(config.CKPT_MOD):
            try:
               
                ckpt = torch.load(config.CKPT_MOD, map_location="cuda", weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                print("load checkpoint")
                del ckpt
                torch.cuda.empty_cache()
            except (RuntimeError, KeyError) as e:
                print(f"Checkpoint incompatible ({e}) — training from scratch.")
    else:    
        print('from scratch')
        
   
            
        
        
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    
    # Datasets
    
    classification_dataset = Vision_Dataset(
       images_folder=r"vg",
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
        images_folder=r"vg",
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
    
    
    caption_dataset = Vision_Dataset(
        images_folder=r"coco\train2014",
        dataset_file=r"VG\gold\caption_coco_train.jsonl",
    )
    caption_dataloader = DataLoader(    
        caption_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    relationships_dataset = Vision_Dataset(
                   images_folder=r"vg",
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
    attributes_dataset = Vision_Dataset(
                 images_folder=r"vg",
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
  
    vqa2_dataset = Vision_Dataset(
                              images_folder=r"coco\val2014",
                              dataset_file=r"VG\gold\vqav2_test_dataset.jsonl",
                          )
    vqa2_dataloader = DataLoader(    
                              vqa2_dataset,
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
        "vqa":vqa2_dataloader,
    }

    weights = {
        "caption": 1,
        "classification": 1,
       "relationship": 1,
        "counting": 1,
        "attributes": 1,
        "vqa":1
    }

    scheduler = MultiTaskBatchScheduler(
        loaders,
        weights
    )
    

   
    
    # تدري
    train_stage3_MoD(model, scheduler, optimizer,criterion,device, scaler,mode="decoders" )
    
    