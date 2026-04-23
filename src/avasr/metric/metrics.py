import jiwer
import re

def compute_wer(references, hypotheses):
    refs_clean = []
    hyps_clean = []
    
    for ref, hyp in zip(references, hypotheses):
        r = re.sub(r'\s+', ' ', str(ref)).strip()
        h = re.sub(r'\s+', ' ', str(hyp)).strip()
                    
        refs_clean.append(r)
        hyps_clean.append(h)
        
    if not refs_clean:
        return 0.0
        
    return jiwer.wer(refs_clean, hyps_clean)