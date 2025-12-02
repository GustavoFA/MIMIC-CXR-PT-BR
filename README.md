# MIMIC-CXR-PT-BR  
A project focused on enhancing the MIMIC-CXR dataset through high-quality Portuguese translations and additional preprocessing tools.

Updates coming soon.


## Models Used

* google/medgemma-27b-text-it → Translation

* google/medgemma-4b-it → Inference & fine-tuning

* BERTimbau Base & XLM-RoBERTa Large → Semantic evaluation (BERTScore)

## Fine-Tuning Configuration (LoRA)

### Main Hyperparameters

| Parameter   | Value |
|-------------|-------|
| α (alpha)   | 16    |
| r           | 16    |
| Dropout     | 0.05  |

### Training Settings

- **Batch size:** 1  
- **Gradient accumulation:** 16  
- **Quantization:** 4-bit  
- **Images per sample:** 2 (VRAM constraint)


### LoRA performance

Evaluation Loss Curve
![Evaluation Loss Curve](GraphResults\eval_loss.png)
Evaluation Entropy Curve
![Evaluation Entropy Curve](GraphResults\eval_entropy.png)
Evaluation Mean Tokens Accurate Curve
![Evaluation Mean Tokens Accurate Curve](GraphResults\mean_tokens_acc.png)

## Evaluation

Two semantic evaluation setups were used:

* Original report (EN) vs. Generated text (PT-BR) – XLM-RoBERTa

* Translated report (PT-BR) vs. Generated text (PT-BR) – BERTimbau

### XLM-RoBERTa

<div style="display: flex; gap: 10px;">

  <div style="text-align: center;">
    <p>Evaluation Precision BERT</p>
    <img src="GraphResults/Precision-bert.jpg" width="300px">
  </div>

  <div style="text-align: center;">
    <p>Evaluation Recall BERT</p>
    <img src="GraphResults/Recall-bert.jpg" width="300px">
  </div>

  <div style="text-align: center;">
    <p>Evaluation F1-Score BERT</p>
    <img src="GraphResults/F1-Score-bert.jpg" width="300px">
  </div>

</div>

### BERTimbau

<div style="display: flex; gap: 10px;">

  <div style="text-align: center;">
    <p>Evaluation Precision BERTimbau</p>
    <img src="GraphResults/Precision-bertimbau.jpg" width="300px">
  </div>

  <div style="text-align: center;">
    <p>Evaluation Recall BERTimbau</p>
    <img src="GraphResults/Recall-bertimbau.jpg" width="300px">
  </div>

  <div style="text-align: center;">
    <p>Evaluation F1-Score BERTimbau</p>
    <img src="GraphResults/F1-Score-bertimbau.jpg" width="300px">
  </div>

</div>


### Summary:

* Few-Shot > Zero-Shot
* LoRA > Few-Shot (on average)
* LoRA shows largest min–max difference, indicating instability