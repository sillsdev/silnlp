from typing import List

from opennmt.utils.wer import sentence_wer


def compute_wer_score(hyps: List[str], refs: List[List[str]]) -> float:
    if len(hyps) == 0:
        return 100.0

    try:
        wer_score = 0.0
        for hyp, ref in zip(hyps, refs[0]):
            wer_score += sentence_wer(ref.lower(), hyp.lower())
        result = wer_score / len(hyps)
    except UnicodeDecodeError:
        print("Unable to compute WER score")
        result = -1
    except ZeroDivisionError:
        print("Cannot divide by zero. Check for empty lines.")
        result = -1

    return result * 100

