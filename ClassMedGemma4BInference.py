import os
import io
import torch
from PIL import Image
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

'''
    Situações que iremos usar dessa classe:
        * FineTuning -> apenas passar imagens e pedir para gerar o texto (podemos testar com EN e PT)
        * MedGemma4B padrão -> esse pode ser destrinchado em vários casos:
            * ZeroShot -> Apenas passar as imagens e pedir para gerar texto (com prompt em PT-BR)
            * FewShot -> Apresentar um exemplo de saída e pedir para fazer o mesmo com as imagens que enviar.
'''


class MedGemma4BInference():

    def __init__(self):
        
        # Modelo base
        self.model_id = 'google/medgemma-4b-it'
        # Acesso ao Hugging Face para obter o modelo
        login()
        # Inicialização do processador e do modelo
        self.load_processor()
        self.load_model()

    def load_processor(self):
        ''' Carregar o processor do MedGemma4B '''
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "right"

    def load_model(self, quantization:bool=True):
        ''' Carregar modelo do MedGemma4B '''
        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if quantization:
            # quantização 4bit
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
        ''' Aplicação do LoRA no modelo '''
        self.model = PeftModel.from_pretrained(self.model, lora_checkpoint_path)

    def processor_data(self, message, images, dtype=None):
        ''' Passar textos e imagens para formato aceito pelo modelo '''
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
        Carrega uma ou mais imagens (a partir de caminho, bytes ou BytesIO)
        e converte todas para RGB (formato necessário para o MedGemma).
        Retorna uma única imagem (PIL.Image) ou uma lista de imagens, conforme o input.
        """
        def _load_single_image(single_input):
            """Carrega uma única imagem e converte para RGB."""
            if isinstance(single_input, (str, os.PathLike)):
                if not os.path.exists(single_input):
                    raise FileNotFoundError(f"O arquivo de imagem '{single_input}' não foi encontrado.")
                return Image.open(single_input).convert('RGB')

            elif isinstance(single_input, (bytes, bytearray, io.BytesIO)):
                if isinstance(single_input, (bytes, bytearray)):
                    single_input = io.BytesIO(single_input)
                return Image.open(single_input).convert('RGB')

            else:
                raise TypeError("Cada elemento deve ser um caminho (str) ou um objeto de arquivo em memória.")

        try:
            # Caso o input seja uma lista ou tupla de imagens
            if isinstance(image_input, (list, tuple)):
                images = []
                for idx, img in enumerate(image_input):
                    try:
                        loaded = _load_single_image(img)
                        images.append(loaded)
                    except Exception as e:
                        print(f"[ERRO] Falha ao carregar imagem {idx + 1}: {e}")
                return images  # lista de imagens

            # Caso seja apenas uma imagem
            else:
                return _load_single_image(image_input)

        except Exception as e:
            print(f"[ERRO] Falha ao carregar imagem(s): {e}")
            return None

    def generate_results(self, text, image_inputs, sys_prompt=None):
        ''' Geração de texto com MedGemma tendo como entrada texto e imagens '''
        # Garante que image_inputs é uma lista
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # Carrega todas as imagens
        images = []
        for img_input in image_inputs:
            img = self.load_image(img_input)
            if img is not None:
                images.append(img)
            else:
                print(f"[ERRO] Falha ao carregar: {img_input}")

        if not images:
            raise ValueError("[ERRO] Nenhuma imagem válida foi carregada. Abortando requisição.")

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

        # Adiciona imagens ao prompt
        pos = 0 if sys_prompt is None else 1
        for img in images:
            message[pos]["content"].append({"type": "image"})

        inputs = self.processor_data(message=message, images=images, dtype=torch.bfloat16)

        # Geração
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        return response
    
if __name__ == "__main__":

    medgemma4b = MedGemma4BInference()

    lora_checkpoint = r'/home/ia368/projetos/fine_tuning/LoRA_saves/medgemma-4b-it-lora-4/checkpoint-240'
    # medgemma4b.apply_lora(lora_checkpoint)

    imgs = [
        '/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test/images/s50008596/2f108c10-c8669b9a-f7f02e0f-272d2904-dd0b345e.jpg.jpg',
        '/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test/images/s50008596/5d7c1542-0e986689-16b380fc-7640a95a-8ef99ac8.jpg.jpg'
    ]
    print(medgemma4b.generate_results('Dê o diagnóstico do raio-x apresentado.', imgs))