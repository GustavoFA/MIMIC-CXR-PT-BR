from bert_score import score
from transformers import AutoTokenizer

class Metrics:
    """
        Collection of utilities for evaluating text generation quality in MedGemma models.
        Provides BERTScore-based metrics for Portuguese and multilingual comparisons.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.max_len = 512

    def _truncate(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len - 1] + [self.tokenizer.sep_token_id]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def bert_score_pt_br(self, text:str, ref:str, verbose:bool=False):
        text = self._truncate(text)
        ref = self._truncate(ref)
        
        prec, rec, f1 = score(
            [text],
            [ref],
            model_type="neuralmind/bert-base-portuguese-cased", # This model has the limitation of 512 tokens
            num_layers=12,
            lang='pt',
            verbose=verbose
        )
        return prec.mean().item(), rec.mean().item(), f1.mean().item()
        
    def bert_score_multlanguage(self, text:str, ref:str, verbose:bool=False):
        prec, rec, f1 = score(
            [text],
            [ref],
            model_type="xlm-roberta-large",
            lang='en',
            verbose=verbose
        )
        return prec.mean().item(), rec.mean().item(), f1.mean().item()
