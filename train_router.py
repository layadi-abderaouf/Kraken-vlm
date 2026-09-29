import torch
import torch.nn as nn
import os
import time
import config
import torch.nn.functional as F
from model.model import Kraken_VLM   
import bitsandbytes as bnb
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import Vision_Dataset,MultiTaskBatchScheduler



@torch.no_grad()
def evaluate_router_phase2(
    model,
    dataloader,
    device,
    tokenizer,
    max_batches=None
):

    model.eval()

    # ================================================================
    # MoD is inside Kraken-vlm
    # ================================================================

    mod = model.decoder

    total_samples = 0

    # ------------------------------------------------
    # Main statistics
    # ------------------------------------------------

    router_loss_oracle_correct = 0
    router_token_acc_oracle_correct = 0

    # ------------------------------------------------
    # Token Accuracy
    # ------------------------------------------------

    total_router_token_acc = 0.0
    total_loss_oracle_token_acc = 0.0
    total_best_token_acc = 0.0

    # ------------------------------------------------
    # Decoder distributions
    # ------------------------------------------------

    router_selection_count = torch.zeros(
        mod.n_decoders,
        dtype=torch.long
    )

    loss_oracle_count = torch.zeros(
        mod.n_decoders,
        dtype=torch.long
    )

    token_acc_oracle_count = torch.zeros(
        mod.n_decoders,
        dtype=torch.long
    )

    # ------------------------------------------------
    # Per-decoder Token Accuracy
    # ------------------------------------------------

    total_decoder_token_acc = torch.zeros(
        mod.n_decoders,
        dtype=torch.float64
    )

    # ------------------------------------------------
    # Loss statistics
    # ------------------------------------------------

    total_router_loss = 0.0
    total_oracle_loss = 0.0
    total_valid_routing_samples = 0

    # ================================================================
    # Evaluation loop
    # ================================================================

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc="Evaluating Router",
            unit="batch"
        )
    ):

        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        (
            image,
            prompt_input_ids,
            prompt_attention_mask,
            target_input_ids,
            target_attention_mask,
            *_
        ) = batch

        # ============================================================
        # Device
        # ============================================================

        image = image.to(device)

        prompt_input_ids = (
            prompt_input_ids.to(device)
        )

        prompt_attention_mask = (
            prompt_attention_mask.to(device)
        )

        target_input_ids = (
            target_input_ids.to(device)
        )

        target_attention_mask = (
            target_attention_mask.to(device)
        )

        # ============================================================
        # VLFM
        # ============================================================

        vlfm_output = model.vlfm(
            image,
            prompt_input_ids,
            prompt_attention_mask
        )

        # ============================================================
        # Teacher forcing
        # ============================================================

        target_input = (
            target_input_ids[:, :-1]
        )

        target_mask = (
            target_attention_mask[:, :-1]
        )

        labels = (
            target_input_ids[:, 1:]
        )

        # ============================================================
        # Analyze router
        # ============================================================

        results = mod.analyze_router(

            vlfm_output=vlfm_output,

            target_ids=target_input,

            images=image,

            target_attention_mask=target_mask,

            prompt_attention_mask=prompt_attention_mask,

            labels=labels,

            pad_token_id=tokenizer.pad_token_id,

            sep_token_id=tokenizer.sep_token_id,

            cls_token_id=tokenizer.cls_token_id
        )

        # ============================================================
        # Batch size
        # ============================================================

        B = image.size(0)

        total_samples += B

        # ============================================================
        # Router == Loss Oracle
        # ============================================================

        router_loss_oracle_correct += (
            results[
                "router_matches_loss_oracle"
            ]
            .sum()
            .item()
        )
        valid_routing_sample = results["valid_routing_sample"]

        num_valid = valid_routing_sample.sum().item()

        total_valid_routing_samples += num_valid

        # ============================================================
        # Router == Token Accuracy Oracle
        # ============================================================

        valid_routing_sample = results["valid_routing_sample"]

        router_matches = results["router_matches_token_acc_oracle"]

        router_token_acc_oracle_correct += (
            (router_matches & valid_routing_sample).sum().item()
        )

        # ============================================================
        # Token Accuracy
        # ============================================================

        router_token_acc = results[
            "router_token_acc"
        ]

        loss_oracle_token_acc = results[
            "loss_oracle_token_acc"
        ]

        token_acc_oracle = results[
            "token_acc_oracle"
        ]

        total_router_token_acc += (
            router_token_acc
            .sum()
            .item()
        )

        total_loss_oracle_token_acc += (
            loss_oracle_token_acc
            .sum()
            .item()
        )

        total_best_token_acc += (
            token_acc_oracle
            .sum()
            .item()
        )

        # ============================================================
        # Per-decoder Token Accuracy
        # ============================================================

        decoder_token_acc = results[
            "decoder_token_acc"
        ]

        total_decoder_token_acc += (
            decoder_token_acc
            .sum(dim=0)
            .double()
            .cpu()
        )

        # ============================================================
        # Decoder indices
        # ============================================================

        router_idx = results[
            "router_idx"
        ]

        loss_oracle_idx = results[
            "loss_oracle_idx"
        ]

        token_acc_oracle_idx = results[
            "token_acc_oracle_idx"
        ]

        # ============================================================
        # Decoder distribution
        # ============================================================

        for decoder_id in range(
            mod.n_decoders
        ):

            router_selection_count[
                decoder_id
            ] += (
                router_idx == decoder_id
            ).sum().cpu()

            loss_oracle_count[
                decoder_id
            ] += (
                loss_oracle_idx == decoder_id
            ).sum().cpu()

            token_acc_oracle_count[
                decoder_id
            ] += (
                token_acc_oracle_idx == decoder_id
            ).sum().cpu()

        # ============================================================
        # Losses
        # ============================================================

        total_router_loss += (
            results[
                "router_loss"
            ]
            .sum()
            .item()
        )

        total_oracle_loss += (
            results[
                "oracle_loss"
            ]
            .sum()
            .item()
        )

    # ================================================================
    # No data
    # ================================================================

    if total_samples == 0:

        print(
            "No samples evaluated."
        )

        return None

    # ================================================================
    # Final metrics
    # ================================================================

    router_loss_oracle_acc = (
        router_loss_oracle_correct
        /
        total_samples
    )

    router_token_acc_oracle_acc = (
        router_token_acc_oracle_correct
        /
        max(total_valid_routing_samples, 1)
    )

    avg_router_token_acc = (
        total_router_token_acc
        /
        total_samples
    )

    avg_loss_oracle_token_acc = (
        total_loss_oracle_token_acc
        /
        total_samples
    )

    avg_best_token_acc = (
        total_best_token_acc
        /
        total_samples
    )

    avg_router_loss = (
        total_router_loss
        /
        total_samples
    )

    avg_oracle_loss = (
        total_oracle_loss
        /
        total_samples
    )

    avg_decoder_token_acc = (
        total_decoder_token_acc
        /
        total_samples
    )

    # ================================================================
    # Print results
    # ================================================================

    print()
    print("=" * 70)
    print("KRAKEN-VLM ROUTER ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal samples: "
        f"{total_samples}"
    )

    print(
        f"\n1. Router == Loss Oracle:"
        f" {router_loss_oracle_acc:.4f}"
        f" ({router_loss_oracle_correct}/"
        f"{total_samples})"
    )

    print(
        f"\n2. Router == Token-Accuracy Oracle:"
        f" {router_token_acc_oracle_acc:.4f}"
        f" ({router_token_acc_oracle_correct}/"
        f"{total_samples})"
    )

    print(
        f"\n3. Router Token Accuracy:"
        f" {avg_router_token_acc:.4f}"
    )

    print(
        f"\n4. Loss-Oracle Token Accuracy:"
        f" {avg_loss_oracle_token_acc:.4f}"
    )

    print(
        f"\n5. Best Decoder Token Accuracy:"
        f" {avg_best_token_acc:.4f}"
    )

    print(
        f"\n6. Average Router Loss:"
        f" {avg_router_loss:.4f}"
    )

    print(
        f"\n7. Average Oracle Loss:"
        f" {avg_oracle_loss:.4f}"
    )

    # ================================================================
    # Decoder distribution
    # ================================================================

    print()
    print("-" * 70)
    print("DECODER DISTRIBUTION")
    print("-" * 70)

    for decoder_id in range(
        mod.n_decoders
    ):

        router_count = (
            router_selection_count[
                decoder_id
            ].item()
        )

        oracle_count = (
            loss_oracle_count[
                decoder_id
            ].item()
        )

        token_acc_oracle_count_i = (
            token_acc_oracle_count[
                decoder_id
            ].item()
        )

        decoder_acc = (
            avg_decoder_token_acc[
                decoder_id
            ].item()
        )

        print(
            f"Decoder {decoder_id}: "
            f"Router={router_count} "
            f"({router_count / total_samples:.4f}) | "
            f"Loss Oracle={oracle_count} "
            f"({oracle_count / total_samples:.4f}) | "
            f"Token Oracle={token_acc_oracle_count_i} "
            f"({token_acc_oracle_count_i / total_samples:.4f}) | "
            f"Token Acc={decoder_acc:.4f}"
        )

    print("=" * 70)

    # ================================================================
    # Return
    # ================================================================

    return {

        "total_samples":
            total_samples,

        "router_loss_oracle_acc":
            router_loss_oracle_acc,

        "router_token_acc_oracle_acc":
            router_token_acc_oracle_acc,

        "avg_router_token_acc":
            avg_router_token_acc,

        "avg_loss_oracle_token_acc":
            avg_loss_oracle_token_acc,

        "avg_best_token_acc":
            avg_best_token_acc,

        "avg_router_loss":
            avg_router_loss,

        "avg_oracle_loss":
            avg_oracle_loss,

        "avg_decoder_token_acc":
            avg_decoder_token_acc,

        "router_selection_count":
            router_selection_count,

        "loss_oracle_count":
            loss_oracle_count,

        "token_acc_oracle_count":
            token_acc_oracle_count
    }
  
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

def train_stage3_router(
    model,
    scheduler,
    optimizer,
    criterion,
    device,
    scaler,
    tokenizer,
    steps=200000,
    log_every=1500,
    save_every=1500,
):
    model.train()

    mod = model.decoder

    # --------------------------------------------------
    # Freeze everything except router
    # --------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    for param in mod.router.parameters():
        param.requires_grad = True

    # --------------------------------------------------
    # Tasks
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Global statistics
    # --------------------------------------------------
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    best_acc = 0.6858

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    pbar = tqdm(
        range(1, steps + 1),
        desc="Training Router",
        unit="step"
    )

    for step in pbar:

        # --------------------------------------------------
        # Get batch
        # --------------------------------------------------
        task, batch = scheduler.get_batch()

        (
            image,
            prompt_input_ids,
            prompt_attention_mask,
            target_input_ids,
            target_attention_mask,
            *_
        ) = batch

        image = image.to(device, non_blocking=True)
        prompt_input_ids = prompt_input_ids.to(
            device,
            non_blocking=True
        )
        prompt_attention_mask = prompt_attention_mask.to(
            device,
            non_blocking=True
        )
        target_input_ids = target_input_ids.to(
            device,
            non_blocking=True
        )
        target_attention_mask = target_attention_mask.to(
            device,
            non_blocking=True
        )

        # --------------------------------------------------
        # Teacher forcing
        # --------------------------------------------------
        target_input = target_input_ids[:, :-1]
        target_mask = target_attention_mask[:, :-1]

        labels = target_input_ids[:, 1:]

        # --------------------------------------------------
        # VLFM
        # Frozen
        # --------------------------------------------------
        with torch.no_grad():

            vlfm_output = model.vlfm(
                image,
                prompt_input_ids,
                prompt_attention_mask
            )

            # --------------------------------------------------
            # Run ALL decoders
            # --------------------------------------------------
            all_logits = mod.forward_all_decoders(
                vlfm_output=vlfm_output,
                target_ids=target_input,
                images=image,
                target_attention_mask=target_mask,
                prompt_attention_mask=prompt_attention_mask
            )

            # --------------------------------------------------
            # Decoder loss
            # [B, D]
            # --------------------------------------------------
            decoder_losses = mod.compute_decoder_losses(
                  all_logits=all_logits,
                  labels=labels,
                  pad_token_id=tokenizer.pad_token_id,
                  sep_token_id=tokenizer.sep_token_id,
                  cls_token_id=tokenizer.cls_token_id
            )

            # --------------------------------------------------
            # Best decoder = lowest loss
            # --------------------------------------------------
            best_decoder_loss = decoder_losses.min(dim=-1).values

            # All decoders that achieved the lowest loss
            is_best_decoder = (
                decoder_losses == best_decoder_loss.unsqueeze(1)
            )

            # --------------------------------------------------
            # Ignore samples where ALL decoders are wrong
            # --------------------------------------------------
            valid_samples = torch.isfinite(best_decoder_loss)
            # --------------------------------------------------
            # Create oracle labels
            # --------------------------------------------------
            task_decoder = DECODER_MAP[task]
            
            oracle_idx = torch.empty(
                decoder_losses.size(0),
                dtype=torch.long,
                device=decoder_losses.device
            )
            
            for i in range(decoder_losses.size(0)):
            
                best_decoders = torch.where(
                    is_best_decoder[i]
                )[0]
            
                # --------------------------------------------------
                # Only one decoder has the best accuracy
                # --------------------------------------------------
                if best_decoders.numel() == 1:
            
                    oracle_idx[i] = best_decoders[0]
            
                # --------------------------------------------------
                # Tie between multiple decoders
                # --------------------------------------------------
                else:
            
                    # Prefer the decoder associated with the task
                    if is_best_decoder[i, task_decoder]:
            
                        oracle_idx[i] = task_decoder
            
                    else:
            
                        # Fallback: deterministic choice
                        # among the tied best decoders
                        oracle_idx[i] = best_decoders[0]

        # --------------------------------------------------
        # Router forward
        # Gradients enabled
        # --------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            router_logits = mod.router(vlfm_output)

            # Only train on samples where at least
            # one decoder produced the correct answer
           
            router_logits_valid = router_logits[valid_samples]
            oracle_idx_valid = oracle_idx[valid_samples]

            
            if task == "caption":
                 # Always route caption samples to decoder 4
                 oracle_idx_valid = torch.full(
                     (router_logits_valid.size(0),),
                     4,
                     dtype=torch.long,
                     device=router_logits_valid.device
                 )
            
            

            if router_logits_valid.numel() > 0:

                loss = F.cross_entropy(
                    router_logits_valid,
                    oracle_idx_valid
                )
            else:
                # No valid samples in this batch
                continue

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------
        batch_size = image.size(0)

        valid_count = valid_samples.sum().item()

        if valid_count > 0:

            router_idx = router_logits_valid.argmax(dim=-1)

            router_correct = (
                router_idx == oracle_idx_valid
            ).sum().item()

            router_acc = (
                router_correct / valid_count
            )

            total_correct += router_correct
            total_valid += valid_count

            total_loss += loss.item()

            # ----------------------------------------------
            # Task statistics
            # ----------------------------------------------
            if task in task_stats:

                task_stats[task]["loss"] += loss.item()

                task_stats[task]["acc"] += router_acc

                task_batches[task] += 1

        # --------------------------------------------------
        # Update tqdm
        # --------------------------------------------------
        if total_valid > 0:

            overall_acc = (
                total_correct / total_valid
            )

        else:
            overall_acc = 0.0

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{overall_acc:.4f}"
        })

        # ==================================================
        # LOG EVERY N STEPS
        # ==================================================
        if step % log_every == 0:

            print("\n")
            print("=" * 70)
            print(f"STEP {step}")
            print("=" * 70)

            # ----------------------------------------------
            # Overall
            # ----------------------------------------------
            if total_valid > 0:

                overall_acc = (
                    total_correct / total_valid
                )

            else:
                overall_acc = 0.0

            print(
                f"Overall Router Accuracy: "
                f"{overall_acc:.4f}"
            )

            print(
                f"Total Valid Samples: "
                f"{total_valid}"
            )

            # ----------------------------------------------
            # Per-task statistics
            # ----------------------------------------------
            print("\nPer-Task Router Accuracy:")

            for task_name in tasks:

                batches = task_batches[task_name]

                if batches == 0:
                    print(
                        f"  {task_name:<15} "
                        f"No samples"
                    )
                    continue

                task_acc = (
                    task_stats[task_name]["acc"]
                    / batches
                )

                task_loss = (
                    task_stats[task_name]["loss"]
                    / batches
                )

                print(
                    f"  {task_name:<15} "
                    f"Acc: {task_acc:.4f} | "
                    f"Loss: {task_loss:.4f} | "
                    f"Batches: {batches}"
                )

            print("=" * 70)

        # ==================================================
        # SAVE CHECKPOINT
        # ==================================================
        if step % save_every == 0:

            # ----------------------------------------------
            # Calculate current overall accuracy
            # ----------------------------------------------
            if total_valid > 0:

                overall_acc = (
                    total_correct / total_valid
                )

            else:
                overall_acc = 0.0

            # ----------------------------------------------
            # Best model
            # ----------------------------------------------
            if overall_acc > best_acc:

                best_acc = overall_acc

                print(
                    f"\nNew best accuracy: "
                    f"{best_acc:.4f}"
                )

                print("save best model")

                save_checkpoint(
                    "MoD-v2-checkpoint-best.pth",
                    model,
                    optimizer,
                    scaler
                )

            # ----------------------------------------------
            # Normal checkpoint
            # ----------------------------------------------
            else:

                print("save model")

                save_checkpoint(
                    "MoD-v2-checkpoint.pth",
                    model,
                    optimizer,
                    scaler
                )

            # ----------------------------------------------
            # Reset statistics
            # ----------------------------------------------
            total_loss = 0.0
            total_correct = 0
            total_valid = 0

            task_stats, task_batches = reset_task_stats(
                tasks
            )

    print("\nTraining finished.")

    return model
    


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
                model.parameters(),        
                lr=1e-4,        
                weight_decay=0.05)

  
    criterion = nn.CrossEntropyLoss()
    #scaler
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume logic ----
    if os.path.exists(config.CKPT_MOD):
            try:
                ckpt = torch.load(config.CKPT_MOD, map_location="cuda", weights_only=True)
               
                model.load_state_dict(ckpt["model_state_dict"])
                #optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                #scaler.load_state_dict(ckpt["scaler_state_dict"])
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
        "caption": 0.5,
        "classification": 1,
       "relationship": 1,
        "counting": 1,
        "attributes": 1,
        "vqa":4
    }

    scheduler = MultiTaskBatchScheduler(
        loaders,
        weights
    )
    
    # تدريب
    #evaluate_router_phase2(model,caption_dataloader,"cuda",tokenizer,200)
    train_stage3_router(model, scheduler, optimizer,criterion,device, scaler,tokenizer )
    #torch.save(model.router.state_dict(), "router.pth")
    