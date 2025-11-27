import os
import io
import torch
from PIL import Image
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


class MedGemma4BInference():
    """
        Inference wrapper for the MedGemma-4B-IT model.
        
        This class handles:
        - Loading the base MedGemma 4B IT model (with optional 4-bit quantization)
        - Loading and applying LoRA checkpoints
        - Preprocessing text & images using the model processor
        - Running inference to generate medical reports or answers
    """

    def __init__(self):
        
        # Base model
        self.model_id = 'google/medgemma-4b-it'
        # Login into Hugging Face (insert your API key here)
        login()
        # Initialize processor and model
        self.load_processor()
        self.load_model()

    def load_processor(self):
        """Load MedGemma4B processor and configure tokenizer padding."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "right"

    def load_model(self, quantization:bool=True):
        """Load MedGemma4B model, optionally using 4-bit quantization."""
        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if quantization:
            # Enable 4-bit quantization using bitsandbytes
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, 
            **model_kwargs
        )

    def apply_lora(self, lora_checkpoint_path:str):
        """Load a LoRA checkpoint and merge it with the base model."""
        self.model = PeftModel.from_pretrained(self.model, lora_checkpoint_path)

    def processor_data(self, message, images, dtype=None):
        """Format text + images into the structure required by MedGemma."""
        input_text = self.processor.apply_chat_template(
            message,
            add_generation_prompt=False,
            tokenize=False,
        ).strip()

        inputs = self.processor(
            text=input_text,
            images=images,
            return_tensors='pt',
            padding=True
        )

        if dtype:
            return inputs.to(self.model.device, dtype=dtype)

        return inputs.to(self.model.device)

    def load_image(self, image_input):
        """
            Loads one or more images (path, bytes, or BytesIO) and converts them to RGB.
            Returns a single image or a list of images depending on the input type.
        """
        def _load_single_image(single_input):
            """Load a single image and convert it to RGB."""
            if isinstance(single_input, (str, os.PathLike)):
                if not os.path.exists(single_input):
                    raise FileNotFoundError(f"Image file '{single_input}' was not found.")
                return Image.open(single_input).convert('RGB')

            elif isinstance(single_input, (bytes, bytearray, io.BytesIO)):
                if isinstance(single_input, (bytes, bytearray)):
                    single_input = io.BytesIO(single_input)
                return Image.open(single_input).convert('RGB')

            else:
                raise TypeError("Each element must be a file path (str) or an in-memory file object.")

        try:
             # List or tuple to load multiple images
            if isinstance(image_input, (list, tuple)):
                images = []
                for idx, img in enumerate(image_input):
                    try:
                        loaded = _load_single_image(img)
                        images.append(loaded)
                    except Exception as e:
                        print(f"[ERROR] Failed to load image {idx + 1}: {e}")
                return images  

            # Single image
            else:
                return _load_single_image(image_input)

        except Exception as e:
            print(f"[ERROR] Failed to load image(s): {e}")
            return None

    def generate_results(self, text, image_inputs, sys_prompt=None):
        """Generate text from MedGemma using text + image inputs."""
        # Force image_inputs to be a list
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # Load all images
        images = []
        for img_input in image_inputs:
            img = self.load_image(img_input)
            if img is not None:
                images.append(img)
            else:
                print(f"[ERROR] Could not load: {img_input}")

        if not images:
            raise ValueError("[ERROR] No valid images were loaded. Aborting request.")

        message = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': text
                    }
                ]
            }
        ]
        if sys_prompt is not None:
            message.insert(0, {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': sys_prompt
                    }
                ]
            })

        # Insert image tokens in the user message
        pos = 0 if sys_prompt is None else 1
        for img in images:
            message[pos]["content"].append({"type": "image"})

        inputs = self.processor_data(message=message, images=images, dtype=torch.bfloat16)

        # Generate response
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        return response
    
if __name__ == "__main__":

    medgemma4b = MedGemma4BInference()

    lora_checkpoint = r'/home/ia368/projetos/fine_tuning/LoRA_saves/medgemma-4b-it-lora-5/checkpoint-240'
    medgemma4b.apply_lora(lora_checkpoint)

    imgs = [
        '/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test/images/s50008596/2f108c10-c8669b9a-f7f02e0f-272d2904-dd0b345e.jpg',
        '/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test/images/s50008596/5d7c1542-0e986689-16b380fc-7640a95a-8ef99ac8.jpg'
    ]
    print(medgemma4b.generate_results('Dê o diagnóstico do raio-x apresentado.', imgs))