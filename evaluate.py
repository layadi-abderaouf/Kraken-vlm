import os
import re
import json
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer


# ==============================================================
# IMPORTS
# ==============================================================

from model.model import VisionMind_V3
from dataset import Vision_Dataset


# ==============================================================
# CONFIG
# ==============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------
# Evaluation results
# ----------------------------------------------------------
RESULTS_FILE = r"D:\datasets\vqa_evaluation_results.jsonl"

# ----------------------------------------------------------
# Dataset settings
# ----------------------------------------------------------
# True:  يعيد جميع الأهداف (لـ VQA v2)
# False: يعيد الهدف الرئيسي فقط (للإجابة الواحدة)
RETURN_TARGETS = True

# True:  وضع VQA v2 (consensus scoring: min(count/3, 1))
# False: وضع الإجابة الواحدة (binary: 1.0 أو 0.0)
VQA_V2_MODE = False

# False: نسمح بتقييم إجابات متعددة الكلمات
ONLY_SINGLE_WORD = False

# ----------------------------------------------------------
# Generation
# ----------------------------------------------------------
MAX_NEW_TOKENS = 3

# token IDs
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102


# ==============================================================
# TOKENIZER
# ==============================================================

tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

# ==============================================================
# MODEL
# ==============================================================

model = VisionMind_V3(vocab_size=tokenizer.vocab_size).to(device)


if os.path.exists("checkpoint.pth"):
            try:
                ckpt = torch.load("checkpoint.pth", map_location="cuda", weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                #model.load_state_dict(
                #    torch.load(
                #        "model-b.pth",
                #         map_location=device,
                #        weights_only=True
                #    )
                #)
                print("load model b")
                del ckpt
                torch.cuda.empty_cache()
            except (RuntimeError, KeyError) as e:
                print(f"Checkpoint incompatible ({e}) — training from scratch.")
else:    
        print('from scratch')
        
model = model.half()


model.eval()


# ==============================================================
# DATASET
# ==============================================================


test_dataset = Vision_Dataset(
    images_folder=r"D:\datasets\coco\val2014",
    dataset_file=r"D:\datasets\VG\gold\pope_test_dataset.jsonl",
    max_length=32,
    return_targets=RETURN_TARGETS,
    only_single_word=ONLY_SINGLE_WORD,
)

# ==============================================================
# COLLATE FUNCTION
# ==============================================================

def collate_fn(batch):
    (
        images,
        prompt_ids,
        prompt_masks,
        target_ids,
        target_masks,
        tasks,
        targets,
        image_names,
        prompts,
    ) = zip(*batch)

    images = torch.stack(images)
    prompt_ids = torch.stack(prompt_ids)
    prompt_masks = torch.stack(prompt_masks)
    target_ids = torch.stack(target_ids)
    target_masks = torch.stack(target_masks)

    targets = list(targets)
    image_names = list(image_names)
    prompts = list(prompts)

    return (
        images,
        prompt_ids,
        prompt_masks,
        target_ids,
        target_masks,
        targets,
        image_names,
        prompts,
    )


# ==============================================================
# DATALOADER
# ==============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=collate_fn,
)

print("=" * 70)
print("VQA ACCURACY EVALUATION")
print("=" * 70)
print(f"Samples          : {len(test_dataset):,}")
print(f"Batch size       : 16")
print(f"Return targets   : {RETURN_TARGETS}")
print(f"VQA v2 mode      : {VQA_V2_MODE}")
print(f"Single word only : {ONLY_SINGLE_WORD}")
print(f"Max new tokens   : {MAX_NEW_TOKENS}")
print("=" * 70)


# ==============================================================
# TEXT NORMALIZATION
# ==============================================================

def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[.,!?;:\"'()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================
# CHECK SINGLE WORD
# ==============================================================

def is_one_word(text):
    words = re.findall(r"\b[\w'-]+\b", str(text))
    return len(words) == 1


# ==============================================================
# ACCURACY FUNCTION (وضعان)
# ==============================================================

def compute_score(prediction, ground_truth_answers, vqa_v2_mode=True):
    """
    حساب الدرجة حسب الوضع:
    
    VQA v2 mode (True):
        - ground_truth_answers: قائمة بجميع إجابات البشر
        - score = min(عدد_المطابقات / 3.0, 1.0)
    
    Single-answer mode (False):
        - ground_truth_answers: قائمة بهدف واحد فقط
        - score = 1.0 إذا تطابقت، 0.0 إذا لم تتطابق
    """
    if not ground_truth_answers:
        return 0.0

    # عدّ المطابقات
    match_count = sum(1 for ans in ground_truth_answers if ans == prediction)

    if vqa_v2_mode:
        # VQA v2: consensus scoring
        return min(match_count / 3.0, 1.0)
    else:
        # Single answer: binary scoring
        return 1.0 if match_count > 0 else 0.0


# ==============================================================
# AUTOREGRESSIVE GENERATION
# ==============================================================

@torch.no_grad()
def generate_batch(images, prompt_ids, prompt_attention_mask, max_new_tokens=8):
    with torch.autocast("cuda", dtype=torch.float16):
       weight_memory = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        ) / 1024**3

       #print("Weight memory:", weight_memory)
       #torch.cuda.empty_cache()
       #torch.cuda.reset_peak_memory_stats()
       batch_size = images.size(0)

       target_ids = torch.full(
           (batch_size, 1), CLS_TOKEN_ID, dtype=torch.long, device=images.device
       )

       decoder_attention_mask = torch.ones(
           batch_size, 1, dtype=torch.bool, device=images.device
       )

       finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

   
       for _ in range(max_new_tokens):
           logits = model(
               images,
               prompt_ids,
               target_ids,
               prompt_attention_mask=prompt_attention_mask,
               decoder_attention_mask=decoder_attention_mask,
           )

           next_token_logits = logits[:, -1, :]
           next_tokens = torch.argmax(next_token_logits, dim=-1)

           next_tokens = torch.where(
               finished,
               torch.full_like(next_tokens, SEP_TOKEN_ID),
               next_tokens,
           )

           target_ids = torch.cat([target_ids, next_tokens.unsqueeze(1)], dim=1)

           new_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=images.device)
           decoder_attention_mask = torch.cat([decoder_attention_mask, new_mask], dim=1)

           finished = finished | (next_tokens == SEP_TOKEN_ID)

           if finished.all():
               break
       #torch.cuda.synchronize()

       #allocated = torch.cuda.max_memory_allocated() / 1024**3
       #reserved = torch.cuda.max_memory_reserved() / 1024**3

       #print("Peak allocated:", allocated)
       #print("Peak reserved:", reserved)

    return target_ids

# ==============================================================
# EVALUATION
# ==============================================================

total_score = 0.0
total = 0

all_predictions = []
all_ground_truths = []


# --------------------------------------------------------------
# Open results file
# --------------------------------------------------------------

results_file = open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
)


with torch.no_grad():

    progress_bar = tqdm(
        test_loader,
        desc="Evaluating"
    )

    for batch in progress_bar:

        (
            images,
            prompt_ids,
            prompt_masks,
            target_ids,
            target_masks,
            targets,
            image_names,
            prompts,
        ) = batch
         # ------------------------------------------------------
        # Move tensors to device
        # ------------------------------------------------------
        
        images = images.to(
                    device,
                    non_blocking=True
                )
        
        prompt_ids = prompt_ids.to(
                    device,
                    non_blocking=True
                )
        
        prompt_masks = prompt_masks.to(
                    device,
                    non_blocking=True
                )
        target_ids = target_ids.to(
                            device,
                            non_blocking=True
                        )
        
        

       
        # ------------------------------------------------------
        # Generate
        # ------------------------------------------------------
        #torch.cuda.synchronize()
        #start_time = time.perf_counter()
         
        generated_ids = generate_batch(
            images=images,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_masks,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        #torch.cuda.synchronize()
        #elapsed_time = time.perf_counter() - start_time

        #print(f"Generation time: {elapsed_time:.4f} sec")
        #batch_size = generated_ids.size(0)
        #throughput = batch_size / elapsed_time

        #print(f"Generation time: {elapsed_time:.4f} sec")
        #print(f"Batch size: {batch_size}")
        #print(f"Throughput: {throughput:.2f} samples/sec")
        #num_tokens = generated_ids.numel()
        #tokens_per_sec = num_tokens / elapsed_time

        #print(f"Tokens/sec: {tokens_per_sec:.2f}")


        # ------------------------------------------------------
        # Decode WITHOUT normalization
        # ------------------------------------------------------

        predictions_raw = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )


        # ------------------------------------------------------
        # Process every sample
        # ------------------------------------------------------

        for i in range(len(predictions_raw)):

            # ==================================================
            # Prediction before normalization
            # ==================================================

            prediction_before_normalize = predictions_raw[i]


            # ==================================================
            # Prediction after normalization
            # ==================================================

            prediction = normalize_answer(
                prediction_before_normalize
            )


            # ==================================================
            # Ground truth
            # ==================================================

            sample_targets = targets[i]


            # If string -> list
            if isinstance(sample_targets, str):
                sample_targets = [sample_targets]


            # Normalize ground truth
            sample_targets = [
                normalize_answer(x)
                for x in sample_targets
            ]


            # ==================================================
            # Single-word filtering
            # ==================================================

            if ONLY_SINGLE_WORD:

                sample_targets = [
                    x
                    for x in sample_targets
                    if is_one_word(x)
                ]

                if len(sample_targets) == 0:
                    continue


            # ==================================================
            # Compute score
            # ==================================================

            score = compute_score(
                prediction,
                sample_targets,
                vqa_v2_mode=VQA_V2_MODE
            )


            # ==================================================
            # Determine correctness
            # ==================================================

            correct = score > 0.0


            # ==================================================
            # Correct answer
            # ==================================================

            if len(sample_targets) == 1:

                correct_answer = sample_targets[0]

            else:

                correct_answer = sample_targets


            # ==================================================
            # Save detailed result
            # ==================================================

            result = {
                "image": image_names[i],

                "question": prompts[i],

                "correct_answer": correct_answer,

                "correct_answer_after_normalize": (
                    sample_targets
                    if len(sample_targets) > 1
                    else sample_targets[0]
                ),

                "prediction_before_normalize":
                    prediction_before_normalize,

                "prediction_after_normalize":
                    prediction,

                "score":
                    float(score),

                "correct":
                    bool(correct),
            }


            results_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )


            # ==================================================
            # Statistics
            # ==================================================

            total += 1
            total_score += score

            all_predictions.append(
                prediction
            )

            all_ground_truths.append(
                sample_targets
            )


        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------

        if total > 0:

            current_acc = (
                100.0 * total_score / total
            )

            progress_bar.set_postfix(
                avg_acc=f"{current_acc:.2f}%"
            )


# --------------------------------------------------------------
# Close results file
# --------------------------------------------------------------

results_file.close()


# ==============================================================
# FINAL RESULTS
# ==============================================================

if total > 0:

    accuracy = (
        100.0 * total_score / total
    )

else:

    accuracy = 0.0


print()

print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Total samples : {total:,}")

print(
    f"Total score   : "
    f"{total_score:.2f} / {total:.2f}"
)

print(
    f"Accuracy      : "
    f"{accuracy:.2f}%"
)

print()

print(
    f"Detailed results saved to:"
)

print(
    RESULTS_FILE
)

print("=" * 70)