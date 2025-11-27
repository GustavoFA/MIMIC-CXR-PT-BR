import os
import re

from typing import Any

from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, Trainer

from huggingface_hub import login

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from ClassDataset import ImageAndStudyDataset


# Path to main dir with train and validate set
_files_folder_path = r''
if not os.path.exists(_files_folder_path):
    _files_folder_path = input('Insert the main files folder:\n')

# creating both datasets
train_dataset = ImageAndStudyDataset(os.path.join(_files_folder_path, 'train'), filter_text=False)
val_dataset = ImageAndStudyDataset(os.path.join(_files_folder_path, 'validate'), filter_text=False)

# ----------------------------------------------------------------------------------
# login on hugging face
login()
# Defining MedGemma4B Processor
model_id = 'google/medgemma-4b-it'
processor_4b = AutoProcessor.from_pretrained(model_id)
# Use right padding to avoid issues during training
processor_4b.tokenizer.padding_side = "right"
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# Defining MedGemma4B Multimodal model - follow google example
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# using quantization
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)
model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
# ----------------------------------------------------------------------------------

# ------------- LoRA configuration --------------------
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear", 
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
# ------------------------------------------------------

# defining collate function
def medgemma_collate_fn(examples: list[dict[str, Any]], processor=processor_4b):
    """
    Custom collate function for preparing multimodal batches for Med-Gemma 4B.

    This function:
    - Extracts text messages and images from each example.
    - Applies the chat template to convert message dictionaries into textual prompts.
    - Uses the MedGemma processor to tokenize text and preprocess images.
    - Creates label tensors for supervised training.
    - Masks out special tokens such as padding, image placeholder tokens,
    and specific unused tokens so they do not contribute to the loss.

    Args:
        examples (list[dict]): A list of dataset samples. Each sample is expected to contain:
            - "image": a list of images corresponding to the example.
            - "messages": a list of dictionaries representing a chat-like conversation.
        processor: MedGemma processor responsible for tokenization and image preprocessing.

    Returns:
        dict: A batch dictionary containing:
            - "input_ids": tokenized text
            - "pixel_values": processed image tensors
            - "attention_mask": attention mask for text
            - "labels": training labels with ignored positions masked to -100
    """
    texts = []
    images = []

    # Iterate over the batch and extract text prompts and images
    for example in examples:
        # Append the image list to the images array (each example contains a list of images)
        images.append(example['image'])

        # Convert the chat-style message into a plain text prompt (no tokenization yet)
        texts.append(processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        ).strip())

    # Use the processor to tokenize the texts and preprocess the images
    # padding=True ensures that text sequences are padded to the same length
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Clone input IDs to create labels for supervised training
    labels = batch["input_ids"].clone()

    # Get token ID corresponding to the <image> placeholder token
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]

    # Ignore padded tokens in the loss
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Ignore special image placeholder tokens in the loss
    labels[labels == image_token_id] = -100

    # Ignore an additional special token (specific to the MedGemma4B tokenizer)
    labels[labels == 262144] = -100

    # Add labels to the batch
    batch["labels"] = labels
    return batch

# ---------------- Train Args -------------------------------
root_lora_save_path = r''
lora_name_file = 'medgemma-4b-it-lora-5'
save_path = os.path.join(root_lora_save_path, lora_name_file)

args = SFTConfig(
    # local to save the lora weights
    output_dir=save_path, 

    # Training and validate
    num_train_epochs=4, 
    per_device_train_batch_size=1, # batch size = 1 (GPU limitation)
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=16, # recommended for the GPU limitation
    gradient_checkpointing=True,
    optim='adamw_torch_fused',
    logging_steps=1,

    # checkpoints (saves)
    save_strategy='steps',  # save of some gradient steps
    save_steps=100,           # number of steps
    save_total_limit=5,    # number of checkpoints on the dir
    save_only_model=True, # save only weights

    # Validation
    eval_strategy='steps', 
    eval_steps=1,

    
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=False,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"],
)

# Training
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    data_collator=medgemma_collate_fn,
)
trainer.train()