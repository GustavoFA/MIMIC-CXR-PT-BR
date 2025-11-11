import os
import json
import random
import pandas as pd
from PIL import Image
from collections import Counter

def split_mimic_files(files_dir:str, 
             json_path:str=None, 
             n_train:int=100000, 
             n_val:int=50000, 
             n_test:int=7 # few data
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

        JSON data example:

        Data example:

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
        df_mimic_set_label = pd.read_csv(os.path.join(files_dir, 'mimic-cxr-2.1.0-test-set-labeled.csv'))
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

def balanced_data(files_dir:str,
                  split_mimic_file:str,
                  n_train:int=10,
                  n_val:int=5,
                  img_res:int=448,
                  filter_text:bool=True,
                  final_file_name:str='MIMIC_BALANCED'
                  ) -> str:
    '''
        Criar um diretório as separações para treino e validação.
        Para cada estudo deve ter apenas um label (o primeiro), baseado no CheXpert
        As imagens devem ser nomeadas com um número igual ao do texto.
        Para os textos temos que obter apenas a parte depois de findings
    '''

    # Carregar o json com as divisões feitas por paciente/estudo
    with open(split_mimic_file, 'r') as f:
        mimic_split = json.load(f)
    # Caminho de saída
    output_dir = os.path.join(os.path.dirname(split_mimic_file), final_file_name)
    os.makedirs(output_dir, exist_ok=True)
    # Percorrer cada split (treino e validação)
    for split_name, split_data in mimic_split.items():
        # Para o caso de teste não temos que tratar o dataset
        if split_name == 'test':
            continue
        
        # número de amostras por split
        n_samples = n_train if split_name == 'train' else n_val
        # diretórios
        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        txt_dir = os.path.join(split_dir, "texts")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        
        # Coleta dos dados baseado nos labels do CheXpert 
        # IMPORTANTE : Um estudo pode ter mais de um label, porém aqui estamos considerando apenas
        # o primeiro label.

        labels_counter = Counter()

        studies = []
        for patient_id, studies_dict in split_data.items():
            for study_id, info in studies_dict.items():
                # verificar se possui algum label do CheXpert
                chex_labels = info.get("chexpert", [])
                # ignorar textos unknown
                if not chex_labels:
                    continue
                else:
                    # considera apenas a primeira label
                    chex_label = chex_labels[0].lower() 

                # verifica se possui imagens 
                img_paths = info.get("images", [])
                if not img_paths:
                    continue  # pular se não há imagens
                
                # obtem textos
                text_path = info.get("study")

                # limita cada label a possuir um número máximo de amostras
                if labels_counter[chex_label] >= n_samples:
                    continue
                labels_counter[chex_label] += 1

                studies.append({
                    "id": study_id,
                    "text": text_path,
                    "images": img_paths,
                    "label": chex_label
                })

        # Embaralhar e cortar
        # random.shuffle(studies)
        # studies = studies[:n_samples]

        # Processar cada estudo
        for idx, study in enumerate(studies, start=1):
            study_id = study["id"]
            # text_file = study["text"]
            text_file = os.path.normpath(study["text"])
            label = study["label"]
            img_files = study["images"]

            # if not os.path.isabs(text_file):
            #     text_file = os.path.join(files_dir, text_file)
            if not os.path.isabs(text_file):
                if text_file.startswith("files" + os.sep):
                    text_file = os.path.join(os.path.dirname(files_dir), text_file)
                else:
                    text_file = os.path.join(files_dir, text_file)
            if not os.path.exists(text_file):
                print(f"[WARN] Texto não encontrado: {text_file}")
                continue

            with open(text_file, "r", encoding="utf-8", errors="ignore") as tf:
                text_content = tf.read()

            # # Ler texto do arquivo
            # try:
            #     with open(text_file, "r", encoding="utf-8", errors="ignore") as tf:
            #         text_content = tf.read()
            # except Exception as e:
            #     print(f"[WARN] Falha ao ler {text_file}: {e}")
            #     continue

            # Filtrar seção após "findings:"
            if filter_text:
                parts = text_content.lower().split("findings:")
                text_content = parts[-1].strip() if len(parts) > 1 else text_content.strip()

            # Criar pasta de imagens do estudo
            study_img_dir = os.path.join(img_dir, study_id)
            os.makedirs(study_img_dir, exist_ok=True)

            # Salvar texto (1 por estudo)
            txt_out = os.path.join(txt_dir, f"{study_id}_{label}.txt")
            with open(txt_out, "w", encoding="utf-8") as out:
                out.write(f"{text_content}")

            # Salvar todas as imagens do estudo
            for img_idx, img_in in enumerate(img_files, start=1):
                img_path = os.path.normpath(img_in)

                if not os.path.isabs(img_path):
                    if img_path.startswith("files" + os.sep):
                        img_path = os.path.join(os.path.dirname(files_dir), img_path)
                    else:
                        img_path = os.path.join(files_dir, img_path)

                if not os.path.exists(img_path):
                    print(f"[WARN] Imagem não encontrada: {img_path}")
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((img_res, img_res))
                    img_out = os.path.join(study_img_dir, f"{os.path.basename(img_path)}.jpg")
                    img.save(img_out)
                except Exception as e:
                    print(f"[WARN] Falha ao processar imagem {img_path}: {e}")

    return f"Processo concluído com sucesso\nNúmero de amostras:\n{labels_counter=}\nDiretório criado em: {output_dir}"

if __name__ == '__main__':

    # # building the mini-mimic-dataset
    # split_mimic_files(
    #     r'C:\IA368\small_mimic_cxr\mimic_small',
    #     r'C:\IA368'
    #     )
    
    print(balanced_data(
       r'C:\IA368\small_mimic_cxr\mimic_small\files',
       r'C:\IA368\MIMIC_SPLIT.json',
       10,
       5,
       224,
       False
    ))

    # print('The code has been started')