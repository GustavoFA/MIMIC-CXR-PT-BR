import os
import json
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ClassMetrics import Metrics
from ClassMedGemma4BInference import MedGemma4BInference


def run_medgemma4b_generation(
        files_path:str, 
        file_results_name:str,
        text_user_input:str,
        text_sys_input:str=None,
        load_LoRA:str=None,
        run_metrics:bool=True,
        save_graph:bool=True,
        ) -> None:
    """
    Runs MedGemma-4B inference on all studies inside a dataset folder, optionally
    evaluates the generated reports using BERTScore (Portuguese[BERTimbau] and English), and
    saves both the JSON results and their corresponding metric plots.

    The dataset is expected to follow this structure:
        files_path/
            images/
                <study_id>/
                    image1.jpg
                    image2.jpg
                    ...
            texts_pt/
                <study_id>_xxx.txt   # Portuguese reference report
            texts_en/
                <study_id>_xxx.txt   # English reference report

    Parameters
    ----------
    files_path : str
        Path to the dataset root containing the images/ and texts_* directories.

    file_results_name : str
        Base name used to save the output JSON and metric figures. A folder with
        this name will be created automatically.

    text_user_input : str
        The user prompt sent to the MedGemma model.

    text_sys_input : str, optional
        Optional system prompt for MedGemma.

    load_LoRA : str, optional
        Path to a LoRA checkpoint. If provided, the model will be loaded with LoRA.

    run_metrics : bool, default=True
        Whether BERTScore metrics should be computed.

    save_graph : bool, default=True
        Whether metric plots should be generated and saved.

    Returns
    -------
    None
        All outputs (JSON + plots) are written to disk.
    """

    # Initialize metrics and MedGemma classes
    metrics = Metrics()
    medgemma4b = MedGemma4BInference()

    if load_LoRA is not None:
        medgemma4b.apply_lora(load_LoRA)

    # Build the directory structure (expected layout)
    '''
    test_files/
        images/
            study_001/
                img1.jpg
                img2.jpg
        texts_pt/
            study_001_xxx.txt
        texts_en/
            study_001_xxx.txt
    '''
    test_dir = Path(files_path)
    images_root = test_dir / "images"
    texts_root_pt = test_dir / "texts_pt"
    texts_root_en = test_dir / "texts_en"

    if not images_root.exists():
        raise RuntimeError("The 'images' directory does not exist in the provided path.")
    
    if not texts_root_pt.exists() or not texts_root_en.exists():
        raise RuntimeError("The 'texts' directory does not exist in the provided path.")
    
    # JSON output
    if ".json" not in file_results_name:
        file_results_name += ".json"

    base_name = file_results_name.replace(".json", "")
    # Below, insert the path for results
    output_dir = Path(r"", base_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / file_results_name
    # -----------------------------------------------------------

    # Load JSON (if it exists)
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    # Metrics
    mean_precision_pt = []
    mean_recall_pt = []
    mean_f1_pt = []

    mean_precision_en = []
    mean_recall_en = []
    mean_f1_en = []

    # Generation
    study_folders = [f for f in images_root.iterdir() if f.is_dir()]
    pbar = tqdm(study_folders, ncols=100)
    for study_folder in pbar:
        pbar.set_description(f"Running MedGemma4B on study: {study_folder.name}")
        if not study_folder.is_dir():
            continue

        images = list(study_folder.glob("*.*"))
        message_result = medgemma4b.generate_results(text_user_input, images, text_sys_input)

        results[study_folder.name] = {
            'text_gen': message_result
        }

        if run_metrics: 
            # Portuguese metrics
            matches_pt = list(texts_root_pt.glob(f"{study_folder.name}_*.txt"))
            if len(matches_pt) == 0:
                print(f"[WARN] No reference text was found for study {study_folder.name}.")
                continue

            with open(matches_pt[0], "r", encoding="utf-8") as f:
                text = f.read()

            results[study_folder.name]['ref_pt'] = text
            precision_pt, recall_pt, f1_score_pt = metrics.bert_score_pt_br(text, message_result)

            mean_precision_pt.append(precision_pt)
            mean_recall_pt.append(recall_pt)
            mean_f1_pt.append(f1_score_pt)

            results[study_folder.name]['bertimbau'] = {
                'p': precision_pt,
                'r': recall_pt,
                'f1': f1_score_pt
            }
            
            # English metrics
            matches_en = list(texts_root_en.glob(f"{study_folder.name}_*.txt"))
            if len(matches_en) == 0:
                print(f"[WARN] No reference text was found for study {study_folder.name}.")
                continue

            with open(matches_en[0], "r", encoding="utf-8") as f:
                text = f.read()

            results[study_folder.name]['ref_en'] = text
            precision_en, recall_en, f1_score_en = metrics.bert_score_multlanguage(text, message_result)

            mean_precision_en.append(precision_en)
            mean_recall_en.append(recall_en)
            mean_f1_en.append(f1_score_en)

            results[study_folder.name]['bert'] = {
                'p': precision_en,
                'r': recall_en,
                'f1': f1_score_en
            }
    
    # Add mean values to the JSON
    if len(mean_f1_pt):
        results['mean_precision_pt'] = sum(mean_precision_pt) / len(mean_precision_pt)
        results['mean_recall_pt'] = sum(mean_recall_pt) / len(mean_recall_pt)
        results['mean_f1_pt'] = sum(mean_f1_pt) / len(mean_f1_pt)

    if len(mean_f1_en):
        results['mean_precision_en'] = sum(mean_precision_en) / len(mean_precision_en)
        results['mean_recall_en'] = sum(mean_recall_en) / len(mean_recall_en)
        results['mean_f1_en'] = sum(mean_f1_en) / len(mean_f1_en)

    # Save JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Generate and save graphs
    if save_graph and metrics:
        #  Portuguese (bertimbau)
        if len(mean_precision_pt):
            x = range(1, len(mean_precision_pt) + 1)
            plt.figure(figsize=(12, 8))
            plt.plot(x, mean_precision_pt, label="Precision", color="#ADD8E6")
            plt.plot(x, mean_recall_pt, label="Recall", color="#FF9999")
            plt.plot(x, mean_f1_pt, label="F1-score", color="#FFFF99")


            plt.axhline(results['mean_precision_pt'], linestyle="--", color="#00008B", label="Mean Precision")
            plt.axhline(results['mean_recall_pt'], linestyle="--", color="#8B0000", label="Mean Recall")
            plt.axhline(results['mean_f1_pt'], linestyle="--", color="#B8860B", label="Mean F1-score")


            plt.xlabel("Samples")
            plt.ylabel("Metric Value")
            plt.title("BERTScore with BERTimbau")
            plt.legend()
            plt.grid()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


            graph_pt = output_dir / f"{base_name}_graph_pt.jpg"
            plt.savefig(graph_pt, format="jpg", dpi=300, bbox_inches="tight")
            plt.close()


        # English-Pt comparison (multilingual BERTScore) 
        if len(mean_precision_en):
            x = range(1, len(mean_precision_en) + 1)
            plt.figure(figsize=(12, 8))
            plt.plot(x, mean_precision_en, label="Precision", color="#ADD8E6")
            plt.plot(x, mean_recall_en, label="Recall", color="#FF9999")
            plt.plot(x, mean_f1_en, label="F1-score", color="#FFFF99")


            plt.axhline(results['mean_precision_en'], linestyle="--", color="#00008B", label="Mean Precision")
            plt.axhline(results['mean_recall_en'], linestyle="--", color="#8B0000", label="Mean Recall")
            plt.axhline(results['mean_f1_en'], linestyle="--", color="#B8860B", label="Mean F1-score")


            plt.xlabel("Samples")
            plt.ylabel("Metric Value")
            plt.title("BERTScore")
            plt.legend()
            plt.grid()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


            graph_en = output_dir / f"{base_name}_graph_en.jpg"
            plt.savefig(graph_en, format="jpg", dpi=300, bbox_inches="tight")
            plt.close()


def run_lora(lora_path:str):
    '''Run inference using a LoRA checkpoint.'''

    lora_name = Path(lora_path)

    run_medgemma4b_generation(
        # insert below the files path
        files_path=r'',
        file_results_name=lora_name.parents[0].name,
        text_user_input='Dê o diagnóstico das imagens apresentadas',
        text_sys_input=None,
        load_LoRA=lora_path,
        run_metrics=True,
        save_graph=True
    )

def run_zero_shot(file_name:str='medgemma4b-it-default-zeroshot'):
    '''Run inference using Zero Shot'''
    
    run_medgemma4b_generation(
        # insert below the files path
        files_path=r'',
        file_results_name=file_name,
        text_user_input='Dê o diagnóstico das imagens apresentadas',
        text_sys_input=None,
        load_LoRA=None,
        run_metrics=True,
        save_graph=True
    )

def run_few_shot(file_name:str='medgemma4b-it-default-fewshot'):
    '''Run inference using Few Shot'''

    text_user_input = '''

    **RELATÓRIO FINAL**

    **INDICAÇÃO:** Drenagem torácica esquerda (tubo de tórax).

    **COMPARAÇÃO:** Radiografia disponível de ___.

    **RADIOGRAFIA DE TÓRAX EM PA (PA - Posteroanterior):**

    O paciente está rotacionado para a direita. Há um novo cateter tipo "pigtail" de toracostomia esquerda terminando na base esquerda, com diminuição do tamanho de um derrame pleural esquerdo de tamanho moderado. Atelectasia adjacente está presente. Derrame pleural e atelectasia à direita pioraram. Não há pneumotórax.

    **IMPRESSÃO:** Tubo de toracostomia na base esquerda, com diminuição intervalar de um derrame esquerdo moderado. Piora da atelectasia e do derrame na base direita.

    Com base no exemplo anterior, dê o diagnóstico das imagens.

    '''
    
    run_medgemma4b_generation(
        # insert below the files path
        files_path=r'',
        file_results_name=file_name,
        text_user_input=text_user_input,
        text_sys_input=None,
        load_LoRA=None,
        run_metrics=True,
        save_graph=True
    )

if __name__ == '__main__':

    # Zero Shot
    run_zero_shot()

    # Few Shot (just one)
    run_few_shot()

    # LoRA
    # Insert below the lora checkpoint path
    run_lora(r'')