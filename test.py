import re

text = '''
## Tradução do Relatório de Raio-X

**RELATÓRIO FINAL**

**EXAME:** TÓRAX (PA e PERFIL)

**INDICAÇÃO:** Histórico: ___F com tosse

**TÉCNICA:** Vistas em PA (póstero-anterior) e perfil do tórax em ortostatismo.

**COMPARAÇÃO:** ___ (Nenhum estudo anterior para comparação)

**ACHADOS:**

A silhueta cardíaca apresenta tamanho discretamente aumentado. A aorta permanece discretamente tortuosa, mas sem alterações em relação a exames anteriores (se houver). Os contornos mediastínicos e hilares estão dentro dos limites normais. A vascularização pulmonar é normal. Não se detecta consolidação focal, derrame pleural ou pneumotórax. Perda de altura leve de um corpo vertebral torácico médio é inalterada.

**IMPRESSÃO:**

Não há anormalidade cardiopulmonar aguda.

---

**Observações:**

*   "___F" provavelmente se refere a uma paciente do sexo feminino (Female), mas a idade ou nome não foi preenchido.
*   "___" na seção de comparação indica que não havia exames anteriores disponíveis para comparação.
*   "Mildly enlarged" foi traduzido como "discretamente aumentado".
*   "Mildly tortuous" foi traduzido como "discretamente tortuosa".
*   "Within normal limits" foi traduzido como "dentro dos limites normais".
*   "Focal consolidation, pleural effusion or pneumothorax" foram traduzidos como "consolidação focal, derrame pleural ou pneumotórax".
*   "Loss of height" foi traduzido como "perda de altura".
*   "Unchanged" foi traduzido como "inalterada" ou "sem alterações".
*   "No acute cardiopulmonary abnormality" foi traduzido como "Não há anormalidade cardiopulmonar aguda".
'''

def text_filter(text: str) -> str:
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


print(text_filter(text))
