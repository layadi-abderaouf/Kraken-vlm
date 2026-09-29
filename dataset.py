import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from transformers import CLIPImageProcessor
from transformers import BertTokenizer
import os
import re
import random





class Vision_Dataset(Dataset):
    def __init__(
        self,
        images_folder,
        dataset_file,
        max_length=32,
        return_targets=False,
        only_single_word=False,
    ):
        self.images_folder = images_folder
        self.max_length = max_length

        # ============================================================
        # CONFIG
        # ============================================================
        self.return_targets = return_targets
        self.only_single_word = only_single_word

        # ============================================================
        # Image processor
        # ============================================================
        self.transform = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        # ============================================================
        # Tokenizer
        # ============================================================
        self.tokenizer = BertTokenizer.from_pretrained(
            "google/bert_uncased_L-4_H-512_A-8"
        )
        # ============================================================
        # Load dataset
        # ============================================================
        self.samples = []
        i = 0
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:

                sample = json.loads(line)
                target = str(sample["target"]).strip()

                # ====================================================
                # Filter: one-word answers only
                # ====================================================

                if self.only_single_word:

                    # استخراج الكلمات الفعلية
                    words = re.findall(r"\b[\w'-]+\b", target)

                    if len(words) != 1:
                        continue
                #if i < 1000:
                self.samples.append(sample)
                i += 1

    # ================================================================
    # Helpers
    # ================================================================
    @staticmethod
    def normalize_text(text):
        """
        توحيد النص للمقارنة:
        - lowercase
        - إزالة المسافات الزائدة
        """

        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)

        return text

    # ================================================================

    def __len__(self):
        return len(self.samples)

    # ================================================================

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_name = sample["image"]
        prompt = sample["prompt"]
        target = sample["target"]

        targets = sample.get("targets", None)

        # ============================================================
        # Image
        # ============================================================

        img_path = os.path.join(
            self.images_folder,
            image_name
        )

        with Image.open(img_path) as image:

            image = image.convert("RGB")

            image = self.transform(
                images=image,
                return_tensors="pt"
            )["pixel_values"].squeeze(0)

        # ============================================================
        # Prompt
        # ============================================================

        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # ============================================================
        # Target
        # ============================================================

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # ============================================================
        # Task
        # ============================================================

        task =  sample["task"] 

        # ============================================================
        # TARGETS LOGIC
        # ============================================================

        if self.return_targets:

            # توجد عدة إجابات صحيحة
            if targets is not None and len(targets) > 0:

                if isinstance(targets, str):
                    targets_out = [targets]
                else:
                    targets_out = list(targets)

            # لا توجد targets -> نستعمل target الأساسي
            else:
                targets_out = [target]

        else:

            # فقط الإجابة الأساسية
            targets_out = target

        # ============================================================
        # Return
        # ============================================================

        return (
            image,
            prompt_encoding["input_ids"].squeeze(0),
            prompt_encoding["attention_mask"].squeeze(0).bool(),

            target_encoding["input_ids"].squeeze(0),
            target_encoding["attention_mask"].squeeze(0).bool(),
            task,
            targets_out,
            image_name,
            prompt,
        )
        
        

class MultiTaskBatchScheduler:

    def __init__(self, loaders, weights):

        self.loaders = loaders
        self.weights = weights

        self.iterators = {
            task: iter(loader)
            for task, loader in loaders.items()
        }

        self.tasks = list(loaders.keys())

    def get_batch(self):

        # اختيار Task حسب الـ weights
        task = random.choices(
            self.tasks,
            weights=[self.weights[t] for t in self.tasks],
            k=1
        )[0]

        # محاولة أخذ batch
        try:
            batch = next(self.iterators[task])

        except StopIteration:

            # انتهى الـ DataLoader
            # نعيد تشغيله
            self.iterators[task] = iter(
                self.loaders[task]
            )

            batch = next(self.iterators[task])

        return task, batch