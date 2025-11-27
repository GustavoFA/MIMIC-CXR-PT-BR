import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

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
      - filter_text (bool): filter the reference text.
      - lim_images (bool): whether to limit the number of images to 2.

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
        sys_prompt: str = "Apresente o diagnóstico das imagens de radiografia, em português brasileiro",
        filter_text:bool=True,
        lim_images:bool=True
    ):

        self.images = self.load_images_from_folder(os.path.join(files_path, 'images'))
        self.reports = self.load_texts_from_folder(os.path.join(files_path, 'texts_pt'))
        self.sys_prompt = sys_prompt
        self.filter_text = filter_text
        self.lim_images = lim_images
        
    def load_texts_from_folder(self, folder:str) -> list:
        ''' Load text reports from the dataset folder. '''
        texts = []
        for filename in sorted(os.listdir(folder)):
            path = os.path.join(folder, filename)
            if filename.endswith(".txt"):
                with open(path, 'r', encoding='utf-8') as f:
                    texts.append(f.read().strip())
        return texts
        
    def load_images_from_folder(self, folder:str) -> list[list]:
        ''' Load images from the dataset folder.'''
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
                    print(f" Error loading {path}: {e}")
            if sub_images:
                all_images.append(sub_images)

        return all_images
    
    def text_filter(self, text: str) -> str:
        """
            Extracts the ACHADOS + IMPRESSÃO sections from a medical report 
            and removes markdown markers (**, ---) and section headers.
        """

        text_lower = text.lower()

        # Locate "ACHADOS"
        match_findings = re.search(r'\bachados\b\s*:', text_lower)
        if not match_findings:
            return ""

        start = match_findings.end()

        # Locate "OBSERVAÇÕES" to use as stopping point
        match_observations = re.search(r'\bobserva[cç][oõ]es\b\s*:', text_lower)
        end = match_observations.start() if match_observations else len(text)

        # Extract the raw content between ACHADOS and OBSERVAÇÕES
        extracted = text[start:end].strip()

        # Patterns to clean markdown symbols and section titles
        cleanup_patterns = [
            r"\*\*",                    # remove ** markdown
            r"^achados\s*:\s*",         # remove 'ACHADOS:'
            r"^impress[aã]o\s*:\s*",    # remove 'IMPRESSÃO:'
            r"---",                     # remove horizontal rule
        ]

        for pattern in cleanup_patterns:
            extracted = re.sub(pattern, "", extracted, flags=re.IGNORECASE | re.MULTILINE)

        # Normalize extra blank lines
        extracted = re.sub(r"\n\s*\n+", "\n\n", extracted).strip()

        return extracted

    def __getitem__(self, idx):
        """
            Returns a single dataset item formatted according to the
            multimodal format expected by the MedGemma processor.

            The "messages" structure follows this logic:
                - Each image is represented as {"type": "image"}.
                - After all images, the system/user prompt is added as text.
                - The medical report is provided as the assistant's response.
        """
        
        report = self.reports[idx]
        # filter text
        if self.filter_text:
            report = self.text_filter(report)

        # Create one {"type": "image"} entry for each image in the study
        images = self.images[idx]
        type_images = [{'type': 'image'} for _ in range(len(images))]
        # limit the number of images (max: 2 images)
        if self.lim_images:
            images = images[:2]
            type_images = type_images[:2]

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
    print('Checking dataset class')

    # insert below the path for train data
    files_path = r''
    train_data = ImageAndStudyDataset(files_path, filter_text=False)
    
    output = train_data[random.randint(0, len(train_data)-1)]

    text = output['label']
    images = output['image']
    message = output['messages']

    print(f'\nCOMPLETE OUTPUT:\n{output=}\n\n')

    print(f'TEXT:{text}\n\n')

    for i, img in enumerate(images):
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i}") # doesn't work :/
        plt.show()