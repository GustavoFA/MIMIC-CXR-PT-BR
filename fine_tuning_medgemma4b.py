import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from typing import Any

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from peft import LoraConfig

from trl import SFTConfig, SFTTrainer

'''
    Importar dados 

    Classe Dataset
    
    Collate_fn
    
    inicialização dos modelos
    
    configuração do LoRA
    
    Argumentos

    Treino

'''


class ImageAndStudyDataset(Dataset):
    """
    A PyTorch dataset that groups multiple radiology images from the same study
    together with their corresponding medical report, formatting the output in
    the multimodal chat structure required by MedGemma4B.

    Each dataset entry represents a single radiology study and provides:
      - images: a list of PIL.Image objects belonging to the study
      - label: the ground-truth medical report associated with that study
      - messages: a chat-style multimodal prompt containing:
            * one {"type": "image"} entry for each image
            * a system/user text prompt
            * the medical report as the assistant response

    The dataset expects the following directory structure:
        files_path/
            images/
                study_001/
                    img1.png
                    img2.png
                    ...
                study_002/
                    ...
            texts/
                report_001.txt
                report_002.txt
                ...

    Input:
      - files_path (str): base directory containing 'images/' and 'texts/'.
      - sys_prompt (str): text instruction included in the user message.

    Output (per item):
      A dictionary with:
        {
            'image': [PIL.Image, ...],
            'label': <string medical report>,
            'messages': <multimodal chat prompt>
        }
    """

    def __init__(
        self,
        files_path:str,
        sys_prompt: str = "Apresente o diagnóstico das imagens de radiografia, em português brasileiro"
    ):
        """

        """
        self.images = self.load_images_from_folder(os.path.join(files_path, 'images'))
        self.reports = self.load_texts_from_folder(os.path.join(files_path, 'texts'))
        self.sys_prompt = sys_prompt
        
    def load_texts_from_folder(self, folder:str) -> list:
        ''' Load texts from Colab environment '''
        texts = []
        for filename in sorted(os.listdir(folder)):
            path = os.path.join(folder, filename)
            if filename.endswith(".txt"):
                with open(path, 'r', encoding='utf-8') as f:
                    texts.append(f.read().strip())
        return texts
        
    def load_images_from_folder(self, folder:str) -> list[list]:
        ''' Load images from Colab environment files '''
        all_images = []

        for subfolder in sorted(os.listdir(folder)):
            subfolder_path = os.path.join(folder, subfolder)

            if not os.path.isdir(subfolder_path):
                continue

            sub_images = []
            for filename in sorted(os.listdir(subfolder_path)):
                path = os.path.join(subfolder_path, filename)
                try:
                    img = Image.open(path)
                    sub_images.append(img)
                except Exception as e:
                    print(f"Error to load {path}: {e}")
            if sub_images:
                all_images.append(sub_images)

        return all_images

    def __getitem__(self, idx):
        """
        Returns a single dataset item formatted according to the
        multimodal format expected by the MedGemma processor.

        The "messages" structure follows this logic:
            - Each image is represented as {"type": "image"}.
            - After all images, the system/user prompt is added as text.
            - The medical report is provided as the assistant's response.
        """
        images = self.images[idx]
        report = self.reports[idx]

        # Create one {"type": "image"} entry for each image in the study
        type_images = [{'type': 'image'} for _ in range(len(images))]

        return {
            'image': images,   # list of PIL images for this study
            'label': report,   # ground-truth medical report
            'messages': [
                {
                    "role": "user",
                    "content": type_images + [
                        {'type': 'text', 'text': self.sys_prompt}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {'type': 'text', 'text': report},
                    ],
                }
            ]
        }

    def __len__(self):
        """Returns the number of radiology studies in the dataset."""
        return len(self.images)



if __name__ == '__main__':
    
    # Path to main dir with train and validate set
    _files_folder_path = input('Insert the main files folder:\n')
    
    # create both datasets
    train_dataset = ImageAndStudyDataset(os.path.join(_files_folder_path, 'train'))
    val_dataset = ImageAndStudyDataset(os.path.join(_files_folder_path, 'validate'))
    
    # ----------------------------------------------------------------------------------
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
    save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.__file__)), 'FineTuningModels', 'medgemma-4b-it-lora-0')
    args = SFTConfig(
        output_dir=save_path,                                    # Directory and Hub repository id to save the model to
        num_train_epochs=1,                                      # Number of training epochs
        per_device_train_batch_size=1,                           # Batch size per device during training
        per_device_eval_batch_size=1,                            # Batch size per device during evaluation
        gradient_accumulation_steps=1,                           # Number of steps before performing a backward/update pass
        gradient_checkpointing=True,                             # Enable gradient checkpointing to reduce memory usage
        optim="adamw_torch_fused",                               # Use fused AdamW optimizer for better performance
        logging_steps=1,                                         # Number of steps between logs
        save_strategy="epoch",                                   # Save checkpoint every epoch
        eval_strategy="steps",                                   # Evaluate every `eval_steps`
        eval_steps=1,                                            # Number of steps between evaluations
        learning_rate=2e-4,                                      # Learning rate based on QLoRA paper
        bf16=True,                                               # Use bfloat16 precision
        max_grad_norm=0.3,                                       # Max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                                       # Warmup ratio based on QLoRA paper
        lr_scheduler_type="linear",                              # Use linear learning rate scheduler
        push_to_hub=False,                                       # Push model to Hub (NÃO PRECISA SER ENVIADO PARA O HUGGING FACE)
        report_to="tensorboard",                                 # Report metrics to tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Set gradient checkpointing to non-reentrant to avoid issues
        dataset_kwargs={"skip_prepare_dataset": True},           # Skip default dataset preparation to preprocess manually
        remove_unused_columns = False,                           # Columns are unused for training but needed for data collator
        label_names=["labels"],                                  # Input keys that correspond to the labels
    )
    # ----------------------------------------------------------------------------------

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
    