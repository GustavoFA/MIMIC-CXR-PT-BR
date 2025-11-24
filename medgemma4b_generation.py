import os
import json
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ClassMetrics import Metrics
from ClassMedGemma4BInference import MedGemma4BInference


'''
    tem que ter a função de zeroshot -> usar com medgemma4b padrão e ajustado

    função fewshot -> acho que vale a pena apenas para o caso do padrão

    salvar resultados em um json para ser plotado futuramente

    JSON:
        {
            estudo: {
                ref_en: str,
                ref_pt: str,
                text_ger: str,
                bert: {
                    p:float,
                    r:float,
                    f1:float
                },
                bertimbau: {
                    p:float,
                    r:float,
                    f1:float
                }
            }
        }
        
'''

def run_medgemma4b_generation(
        files_path:str, 
        file_results_name:str,
        text_user_input:str,
        text_sys_input:str=None,
        load_LoRA:str=None,
        run_metrics:bool=True,
        save_graph:bool=True,
        ) -> None:
    
    metrics = Metrics()
    medgemma4b = MedGemma4BInference()

    if load_LoRA is not None:
        medgemma4b.apply_lora(load_LoRA)

    # Verificar se o caminho dos arquivos existe 
    test_dir = Path(files_path)
    images_root = test_dir / "images"
    texts_root = test_dir / "texts"

    if not images_root.exists():
        raise RuntimeError("A pasta 'images' não existe dentro de test/")
    
    if not texts_root.exists():
        raise RuntimeError("A pasta 'texts' não existe dentro de test/")
    
    # -----------------------------------------------------------
    # Criar pasta de saída com o nome do file_results_name
    # -----------------------------------------------------------
    if ".json" not in file_results_name:
        file_results_name += ".json"

    base_name = file_results_name.replace(".json", "")
    output_dir = Path("/home/ia368/projetos/fine_tuning/MetricsResults", base_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Caminho final do JSON
    results_file = output_dir / file_results_name
    # -----------------------------------------------------------

    # Carregar JSON existente, se houver
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    mean_precision = []
    mean_recall = []
    mean_f1 = []

    # percorre cada estudo dentro de images/
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
            matches = list(texts_root.glob(f"{study_folder.name}_*.txt"))
            if len(matches) == 0:
                print(f"[AVISO] Nenhum texto encontrado para estudo {study_folder.name}")
                continue

            with open(matches[0], "r", encoding="utf-8") as f:
                text = f.read()

            results[study_folder.name]['ref_pt'] = text
            precision, recall, f1_score = metrics.bert_score_pt_br(text, message_result)

            mean_precision.append(precision)
            mean_recall.append(recall)
            mean_f1.append(f1_score)

            results[study_folder.name]['bertimbau'] = {
                'p': precision,
                'r': recall,
                'f1': f1_score
            }

    if len(mean_f1):
        results['mean_precision'] = sum(mean_precision) / len(mean_precision)
        results['mean_recall'] = sum(mean_recall) / len(mean_recall)
        results['mean_f1'] = sum(mean_f1) / len(mean_f1)

    # -----------------------------------------------------------
    # Salvar JSON dentro da pasta específica
    # -----------------------------------------------------------
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # -----------------------------------------------------------
    # Salvar gráfico dentro da mesma pasta
    # -----------------------------------------------------------
    if save_graph and metrics and len(mean_precision):

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        x = range(1, len(mean_precision) + 1)

        plt.figure(figsize=(12, 8))

        # Curvas
        plt.plot(x, mean_precision, label="Precision", color="#ADD8E6")
        plt.plot(x, mean_recall, label="Recall", color="#FF9999")
        plt.plot(x, mean_f1, label="F1-score", color="#FFFF99")

        # Linhas horizontais das médias
        plt.axhline(results["mean_precision"], linestyle="--", label="Mean Precision", color="#00008B")
        plt.axhline(results["mean_recall"], linestyle="--", label="Mean Recall", color="#8B0000")
        plt.axhline(results["mean_f1"], linestyle="--", label="Mean F1-score", color="#B8860B")

        plt.xlabel("Samples")
        plt.ylabel("Metric Value")
        plt.title("Precision, Recall, F1-score")
        plt.legend()
        plt.grid()

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Caminho final do gráfico dentro da pasta
        graph_path = output_dir / f"{base_name}_graph.jpg"

        plt.savefig(graph_path, format="jpg", dpi=300, bbox_inches="tight")
        plt.close()


def run_lora(lora_path:str):

    lora_name = Path(lora_path).name

    run_medgemma4b_generation(
        files_path=r'/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test',
        file_results_name=lora_name,
        text_user_input='Dê o diagnóstico das imagens apresentadas',
        text_sys_input=None,
        load_LoRA=lora_path,
        run_metrics=True,
        save_graph=True
    )

def run_zero_shot(file_name:str='medgemma4b-it-default-zeroshot'):
    
    run_medgemma4b_generation(
        files_path=r'/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test',
        file_results_name=file_name,
        text_user_input='Dê o diagnóstico das imagens apresentadas',
        text_sys_input=None,
        load_LoRA=None,
        run_metrics=True,
        save_graph=True
    )

def run_few_shot(file_name:str='medgemma4b-it-default-fewshot'):

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
        files_path=r'/home/ia368/projetos/fine_tuning/MIMIC-DATA-PROCESS/test',
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

    # # Few Shot (just one)
    # run_few_shot()

    # # LoRA
    # run_lora()