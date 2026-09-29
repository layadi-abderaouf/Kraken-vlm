import torch
from PIL import Image
from model.model import Kraken_VLM
import os
import random
from transformers import BertTokenizer
from transformers import CLIPImageProcessor
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import config



# ---- إعداد الجهاز ----
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)

model = Kraken_VLM(vocab_size=tokenizer.vocab_size).to(device)

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters:    {total_params - trainable_params:,}")

if os.path.exists(config.CKPT_MOD):
    try:
        ckpt = torch.load(config.CKPT_MOD, map_location="cuda", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        
        print("load model")
        del ckpt
    except RuntimeError:
       print("Saved weights incompatible — training from scratch.")
else :
    print('from scratch')


model.eval() 


# ---- تحويل الصورة إلى tensor ----
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    processor = CLIPImageProcessor.from_pretrained(
               config.CLIP_MODEL
            )
    img = processor(
        images=img,
        return_tensors="pt"
    )
    return img["pixel_values"]

# ---- توليد الكابتشن ----
def generate(model, image_path, prompt,max_len=32,task_id=0):
    model.eval()
    image = load_image(image_path).to(device)
    ids = [tokenizer.cls_token_id]
    prompt = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

    prompt_ids = prompt["input_ids"].to(device)
    prompt_mask = prompt["attention_mask"].bool().to(device)
    task_id = torch.tensor( [task_id],dtype=torch.long,device=device)
    

    # ---- Encoder ----
    with torch.no_grad():
        while len(ids) < max_len:
            ids_tensor = torch.tensor(
                [ids],
                dtype=torch.long,
                device=device
            )
            decoder_mask = torch.ones_like(
                ids_tensor,
                dtype=torch.bool
            )            
            logits = model(image,prompt_ids,ids_tensor,prompt_mask,decoder_mask)
            next_token_logits = logits[:, -1, :]  
            next_id = torch.argmax(next_token_logits, dim=-1).item()
            # إذا وصلنا إلى نهاية الجملة نتوقف
            if next_id == tokenizer.sep_token_id:
               break

            ids.append(next_id)


    caption = tokenizer.decode(
        ids,
        skip_special_tokens=True
    )
    return caption




def beam_generate(
    model,
    image_path,
    prompt,
    max_len=32,
    beam_size=3,
    image=None,
    task_id=0
):
    model.eval()
    if image == None:
       image = load_image(image_path).to(device)
    else:
        image = image.to(device)   
    prompt = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )
    prompt_ids = prompt["input_ids"].to(device)
    prompt_mask = prompt["attention_mask"].bool().to(device)
    task_id = torch.tensor( [task_id],dtype=torch.long,device=device)

    with torch.no_grad():
        beams = [ ( [tokenizer.cls_token_id],  0.0   ) ]
        finished = []
        for _ in range(max_len):
            candidates = []
            for ids, score in beams:
                if ids[-1] == tokenizer.sep_token_id:
                    finished.append((ids, score))
                    continue
                ids_tensor = torch.tensor(
                    [ids],
                    dtype=torch.long,
                    device=device
                )
                decoder_mask = torch.ones_like(
                    ids_tensor,
                    dtype=torch.bool
                )
                logits = model(
                    image,
                    prompt_ids,
                    ids_tensor,
                    prompt_mask,
                    decoder_mask,
                )
                logits = logits[:, -1, :]
                log_probs = F.log_softmax(
                    logits,
                    dim=-1
                )
                values, indices = torch.topk(
                    log_probs,
                    beam_size,
                    dim=-1
                )
                for k in range(beam_size):
                    next_id = indices[0, k].item()
                    next_score = score + values[0, k].item()
                    candidates.append(
                        (
                            ids + [next_id],
                            next_score
                        )
                    )
            if len(candidates) == 0:
                break

            candidates = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )
            beams = candidates[:beam_size]
        if len(finished) == 0:
            finished = beams
        best_ids, best_score = max(
            finished,
            key=lambda x: x[1] / len(x[0])      # Length Normalization
        )
    return tokenizer.decode(
        best_ids,
        skip_special_tokens=True
    )

# ---- مثال ----
image_path = ""
while True:
   print("enter your question")
   prompt = input()
   
   caption = generate(model,image_path,prompt)
   
   

   
   

