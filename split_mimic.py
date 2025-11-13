import os
import json
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import Counter

def split_mimic_files(files_dir:str, 
             json_path:str=None, 
             n_train:int=None, 
             n_val:int=None, 
             n_test:int=None # few data
             ) -> dict:
    '''
        Function to split the MIMIC-CXR data into training, validation, and test sets. 
        The split is performed by subject and study. For each study, the function stores 
        the path to the medical report (text file), image paths (JPG), and the corresponding 
        CheXpert and NegBio labels.

        :param str files_dir: Path to the main directory containing the MIMIC-CXR files.
        :param str json_path: Path to save the final JSON file.
        :param int n_train: Number of cases for training (each study counts as one case).
        :param int n_val: Number of cases for validation.
        :param int n_test: Number of cases for testing. 
                        (Note: there are only 7 studies in the test set).
        
        :return: Study splits organized by patient (subject).
        :rtype: dict

        --------------------------------------------

        Final JSON data example:

        'train' : {
            'p10000032': {
                's50414267': {
                    'study': all_path\study.txt,
                    'images': [
                        all_path\image1.jpg,
                        all_path\image2.jpg
                    ],
                    'chexpert': [
                        label1,
                        label2
                    ],
                    'negbio': [
                        label1,
                        label2
                    ]
                }
            }
        }
    '''

    # dict to store data and specifications
    files_dict = {
        'train' : {},
        'validate' : {},
        'test' : {}
    }
    # check files path
    if not os.path.exists(files_dir):
        raise ValueError('Invalid path')
    else:
        df_split = pd.read_csv(os.path.join(files_dir, 'mimic-cxr-2.0.0-split.csv'))
        df_chexpert = pd.read_csv(os.path.join(files_dir, 'mimic-cxr-2.0.0-chexpert.csv'))
        df_negbio = pd.read_csv(os.path.join(files_dir, 'mimic-cxr-2.0.0-negbio.csv'))  
    # files folder name
    dir_files = r'files'
    # split limit
    limit = {
        'train' : n_train,
        'validate': n_val,
        'test': n_test
    }
    # count split
    count = {
        'train' : 0,
        'validate': 0,
        'test': 0
    }

    # Read split CSV
    for _, row in df_split.iterrows():
        if (n_train is not None) and (n_val is not None) and (n_test is not None):
            if (count['train'] + count['validate'] + count['test']) >= (n_train + n_val + n_test):
                break   
            split = row['split']
            if count[split] >= limit[split]:
                continue
        sub_id = str(row['subject_id'])
        study_id = row['study_id']
        image_id = row['dicom_id']

        sub_id = f'p{sub_id}'
        study_id = f's{study_id}'

        txt_path = os.path.join(dir_files, sub_id[:3], sub_id, f'{study_id}.txt')
        image_path = os.path.join(dir_files, sub_id[:3], sub_id, study_id, f'{image_id}.jpg')
        
        # Create the split dict
        split_dict = files_dict[split]

        # Create the subject key
        if sub_id not in split_dict:
            split_dict[sub_id] = {} 

        # Create the study key
        if study_id not in split_dict[sub_id]:
            count[split] += 1
            split_dict[sub_id][study_id] = {
                'study': txt_path,
                'images': []
            }

        # Add image
        split_dict[sub_id][study_id]['images'].append(image_path)  
    
    # labels
    chexpert_negbio_labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices"
    ]


    # Read cheXpert
    for _, row in df_chexpert.iterrows():
        sub_id = f"p{str(int(row['subject_id']))}"
        study_id = f"s{str(int(row['study_id']))}"
        for split in files_dict:
            if sub_id in files_dict[split] and study_id in files_dict[split][sub_id]:
                labels = []
                for label in chexpert_negbio_labels:
                    if row[label] == 1.0:
                        labels.append(label)
                files_dict[split][sub_id][study_id]['chexpert'] = labels

    # Read NegBio
    for _, row in df_negbio.iterrows():
        sub_id = f"p{str(int(row['subject_id']))}"
        study_id = f"s{str(int(row['study_id']))}"
        for split in files_dict:
            if sub_id in files_dict[split] and study_id in files_dict[split][sub_id]:
                labels = []
                for label in chexpert_negbio_labels:
                    if row[label] == 1.0:
                        labels.append(label)
                files_dict[split][sub_id][study_id]['negbio'] = labels

    # Save JSON file
    if json_path is None or not os.path.exists(json_path):
        # Save in the Python script dir
        json_path = os.path.join(os.path.dirname(__file__), 'MIMIC_SPLIT.json')
    else:
        json_path = os.path.join(json_path, 'MIMIC_SPLIT.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(files_dict, f, indent=2, ensure_ascii=False)
    print(f'JSON file saved on {json_path}')
    
    return files_dict

def generate_balanced_data(files_dir: str,
                           split_mimic_file: str,
                           n_train: int = 10,
                           n_val: int = 5,
                           img_res: int = 896,
                           filter_text: bool = True,
                           final_file_name: str = 'MIMIC_BALANCED'
                           ) -> str:
    '''
        Generate a balanced dataset with train and validation splits 
        based on a pre-defined MIMIC-CXR patient/study separation file.

        Each study is assigned to a single label (the first CheXpert label available),
        ensuring that each label has up to `n_train` or `n_val` samples for its respective split.
        Corresponding study images and report texts are saved in structured directories.

        :param str files_dir: Path to the main directory containing the MIMIC-CXR image and text files.
        :param str split_mimic_file: Path to the JSON file containing predefined patient/study splits.
        :param int n_train: Maximum number of samples per label to include in the training split.
        :param int n_val: Maximum number of samples per label to include in the validation split.
        :param int img_res: Target image resolution (in pixels) used to resize all images to square dimensions.
        :param bool filter_text: If True, only the section after "Findings:" in the reports will be kept.
        :param str final_file_name: Name of the output directory where the processed dataset will be saved.
        
        :return: Summary string containing label distribution and the output directory path.
        :rtype: str
    '''

    with open(split_mimic_file, 'r') as f:
        mimic_split = json.load(f)

    output_dir = os.path.join(os.path.dirname(split_mimic_file), final_file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over train/val splits with tqdm
    for split_name, split_data in tqdm(mimic_split.items(), desc="Processing splits"):
        if split_name == 'test':
            continue

        n_samples = n_train if split_name == 'train' else n_val

        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        txt_dir = os.path.join(split_dir, "texts")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        labels_counter = Counter()
        studies = []

        # Build study list
        for patient_id, studies_dict in split_data.items():
            for study_id, info in studies_dict.items():
                chex_labels = info.get("chexpert", [])
                if not chex_labels:
                    continue
                chex_label = chex_labels[0].lower()
                img_paths = info.get("images", [])
                if not img_paths:
                    continue
                text_path = info.get("study")
                if labels_counter[chex_label] >= n_samples:
                    continue
                labels_counter[chex_label] += 1
                studies.append({
                    "id": study_id,
                    "text": text_path,
                    "images": img_paths,
                    "label": chex_label
                })

        # Process each study with progress bar
        for study in tqdm(studies, desc=f"Processing {split_name} studies", leave=False):
            study_id = study["id"]
            text_file = os.path.normpath(study["text"])
            label = study["label"]
            img_files = study["images"]

            if not os.path.isabs(text_file):
                if text_file.startswith("files" + os.sep):
                    text_file = os.path.join(os.path.dirname(files_dir), text_file)
                else:
                    text_file = os.path.join(files_dir, text_file)
            if not os.path.exists(text_file):
                tqdm.write(f"[WARN] Text file not found: {text_file}")
                continue

            with open(text_file, "r", encoding="utf-8", errors="ignore") as tf:
                text_content = tf.read()

            # Extract text after "Findings:" if requested
            # TODO - It's not working good :/ 
            if filter_text:
                parts = text_content.lower().split("findings:")
                text_content = parts[-1].strip() if len(parts) > 1 else text_content.strip()

            # Create image directory for the study
            study_img_dir = os.path.join(img_dir, study_id)
            os.makedirs(study_img_dir, exist_ok=True)

            # Save processed text
            txt_out = os.path.join(txt_dir, f"{study_id}_{label}.txt")
            with open(txt_out, "w", encoding="utf-8") as out:
                out.write(f"{text_content}")

            # Process images with inner tqdm (optional)
            for img_in in tqdm(img_files, desc=f"Images for {study_id}", leave=False):
                img_path = os.path.normpath(img_in)
                if not os.path.isabs(img_path):
                    if img_path.startswith("files" + os.sep):
                        img_path = os.path.join(os.path.dirname(files_dir), img_path)
                    else:
                        img_path = os.path.join(files_dir, img_path)

                if not os.path.exists(img_path):
                    tqdm.write(f"[WARN] Image not found: {img_path}")
                    continue
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((img_res, img_res))
                    img_out = os.path.join(study_img_dir, f"{os.path.basename(img_path)}.jpg")
                    img.save(img_out)
                except Exception as e:
                    tqdm.write(f"[WARN] Failed to process image {img_path}: {e}")

    return f"Process completed successfully\nNumber of samples:\n{labels_counter=}\nDirectory created at: {output_dir}"

if __name__ == '__main__':

    # print('Running the split mimic files -> generating JSON split')
    # split_mimic_files(
    #     r'C:\IA368\small_mimic_cxr\mimic_small',
    #     r'C:\IA368'
    # )

    # print('Running generate balanced data')
    # print(generate_balanced_data(
    #    r'C:\IA368\small_mimic_cxr\mimic_small\files',
    #    r'C:\IA368\MIMIC_SPLIT.json',
    #    10,
    #    5,
    #    224,
    #    True
    # ))