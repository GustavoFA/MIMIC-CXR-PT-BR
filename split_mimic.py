import os
import json
import pandas as pd

def split_mimic_files(files_dir:str, 
             json_path:str=None, 
             n_train:int=1000, 
             n_val:int=500, 
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

if __name__ == '__main__':

    # building the mini-mimic-dataset
    split_mimic_files(
        r'C:\IA368\small_mimic_cxr\mimic_small',
        r'C:\IA368'
        )

