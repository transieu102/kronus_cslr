# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

#from utils import Rouge,sacrebleu
import numpy as np

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4

import re

def normalize_gloss_sequence(sequence):
    """
    Apply the same preprocessing steps as the dataset authors' script to normalize gloss sequences.

    Args:
        sequence (str): The raw gloss sequence.
    
    Returns:
        str: The cleaned and normalized gloss sequence.
    """

    # 🔹 Step 1: Remove specific prefixes
    sequence = re.sub(r'\b(loc-|cl-|qu-|poss-|lh-)', '', sequence)

    # 🔹 Step 2: Replace specific words with correct forms
    replacements = {
        'S0NNE': 'SONNE',
        'HABEN2': 'HABEN',
        '__EMOTION__': '',
        '__PU__': '',
        '__LEFTHAND__': '',
        '__EPENTHESIS__': '',
        '+': ' '
        
    }
    for key, value in replacements.items():
        sequence = sequence.replace(key, value)

    # 🔹 Step 3: Merge specific multi-word glosses
    sequence = re.sub(r'\bWIE AUSSEHEN\b', 'WIE-AUSSEHEN', sequence)
    sequence = re.sub(r'\bZEIGEN\b', 'ZEIGEN-BILDSCHIRM', sequence)

    # 🔹 Step 4: Handle specific phrase combinations
    sequence = re.sub(r'\b([A-Z]) ([A-Z][+ ]+)', r'\1+\2', sequence)
    sequence = re.sub(r'[ +]([A-Z]) ([A-Z]) ', r' \1+\2 ', sequence)
    sequence = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', sequence)
    sequence = re.sub(r'([ +]SCH) ([A-Z][ +])', r'\1+\2', sequence)
    sequence = re.sub(r'([ +]NN) ([A-Z][ +])', r'\1+\2', sequence)
    sequence = re.sub(r'([ +][A-Z]) (NN[ +])', r'\1+\2', sequence)
    sequence = re.sub(r'([ +][A-Z]) ([A-Z])$', r'\1+\2', sequence)

    # 🔹 Step 5: Remove specific suffixes (e.g., `RAUM`)
    sequence = re.sub(r'([A-Z][A-Z])RAUM\b', r'\1', sequence)

    # 🔹 Step 6: Remove unwanted markers like "-PLUSPLUS"
    sequence = re.sub(r'-PLUSPLUS', '', sequence)

    # 🔹 Step 7: Remove duplicated consecutive words
    sequence = re.sub(r'\b(\w+)\s+\1\b', r'\1', sequence)

    # 🔹 Step 8: Remove extra spaces
    sequence = re.sub(r'\s+', ' ', sequence).strip()

    return sequence

def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        # print("r: ", r)
        # print("h: ", h)
        # input()
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]
    # print("total_ref_len: ", total_ref_len)
    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del": del_rate,
        "ins": ins_rate,
        "sub": sub_rate,
    }


def wer_single(r, h):
    # print("debug==", r, h)
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)
    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }

def wer_single_list(r, h):
    # print("debug==", r, h)
    # r = r.strip().split()
    # h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)
    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "wer": num_err/num_ref * 100,
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d



def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )