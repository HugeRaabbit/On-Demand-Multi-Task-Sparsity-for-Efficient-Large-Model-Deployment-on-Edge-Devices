#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import random
from copy import deepcopy

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from nltk.corpus import wordnet as wn
# For the first run, uncomment the following two lines to download the WordNet corpus.
# import nltk
# nltk.download('wordnet'); nltk.download('omw-1.4')

from skip_utils.utils import block_remove_qwen2vl

random.seed(42)

# Configuration and Paths
MODEL_PATH = "(path to Qwen2-VL-2B-Instruct)"
DEVICE = "cuda:0"

TOTAL_BLOCKS = 28
REMOVAL_RATIOS = [0.9]          # Maximum Deletion Attempt Ratio
ACC_THRESHOLD_RATIO = 0.90      # Accuracy threshold (relative to baseline) for candidate acceptance

TASK_ORDER = ["traffic light", "car", "obstacle", "person", "bicycle"]
TL_ROOT = "path to traffic light dataset"
CODA_ROOT = "path to CODA dataset"

def is_prediction_correct_wordnet(pred_token: str, category_word: str) -> bool:
    """Checks semantic similarity using WordNet to allow robust label matching."""
    p = pred_token.lower().strip()
    c = category_word.lower().strip()

    pred_synsets = wn.synsets(p, pos=wn.NOUN)
    cat_synsets = wn.synsets(c, pos=wn.NOUN)
    if not pred_synsets or not cat_synsets:
        return False
    for ps in pred_synsets:
        for cs in cat_synsets:
            if ps == cs:
                return True
            sim = ps.wup_similarity(cs)
            if sim is not None and sim > 0.75:
                return True
    return False

def clean_model_response_to_token(response: str) -> str:
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1]
    token = response.strip().split()[0] if response.strip() else ""
    token = token.lower().strip().strip(",.?!;:\"'`")
    return token

def make_Qwen2_inputs(processor, image, prompt, device):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    ).to(device)
    return inputs

def priority_rank_for_task(block_idx: int, task_name: str, ref_lists: list):
    """
    Returns a sortable priority key where lower values are prioritized.
    """
    if task_name == "traffic light" or not ref_lists:
        return 1

    if task_name in ("car", "bicycle"):
        ref = set(ref_lists[0]) if len(ref_lists) >= 1 else set()
        return 0 if block_idx in ref else 1

    if task_name in ("obstacle", "person"):
        ref1 = set(ref_lists[0]) if len(ref_lists) >= 1 else set()
        ref2 = set(ref_lists[1]) if len(ref_lists) >= 2 else set()
        in1, in2 = (block_idx in ref1), (block_idx in ref2)
        if in1 and in2:
            return 0
        if in1 and (not in2):
            return 1
        if (not in1) and in2:
            return 2
        return 3

    return 10

def greedy_select_for_one_task(task_name: str,
                               model,
                               processor,
                               total_blocks: int,
                               removal_ratio_limit: float,
                               baseline_acc: float,
                               acc_threshold_ratio: float,
                               log_file,
                               ref_lists: list):
    max_remove = int(total_blocks * removal_ratio_limit)
    removal_list = []

    log_file.write(f"[TASK] {task_name}\n")
    log_file.write(f"  Baseline Acc: {baseline_acc:.2f} | Threshold: {baseline_acc * acc_threshold_ratio:.2f}\n")
    log_file.write(f"  Reference lists (by priority): {ref_lists}\n")
    log_file.write("-" * 60 + "\n")
    log_file.flush()

    def eval_fn(removal):
        if task_name == "traffic light":
            return evaluate_traffic_light(removal, model, processor, DEVICE, MAX_NEW_TOKENS[task_name])
        else:
            return evaluate_coda_single_category(task_name, removal, model, processor, DEVICE, MAX_NEW_TOKENS[task_name])

    while len(removal_list) < max_remove:
        candidates = [i for i in range(total_blocks) if i not in removal_list]
        per_cand_acc = {}
        per_cand_pri = {}

        step_id = len(removal_list) + 1
        pbar = tqdm(candidates, desc=f"[{task_name}] Step {step_id} (|R|={len(removal_list)})")

        for block in pbar:
            test_list = removal_list + [block]
            acc = eval_fn(test_list.copy())
            pri = priority_rank_for_task(block, task_name, ref_lists)
            per_cand_acc[block] = acc
            per_cand_pri[block] = pri
            log_file.write(
                f"Try add layer={block:02d} | priority={pri} | acc={acc:.2f} | "
                f"keep_if >= {baseline_acc*acc_threshold_ratio:.2f} | test_list={test_list}\n"
            )
            log_file.flush()

        feasible = [b for b in candidates if per_cand_acc[b] >= baseline_acc * acc_threshold_ratio]
        if not feasible:
            log_file.write(f"STOP: No candidate keeps acc >= {baseline_acc*acc_threshold_ratio:.2f}.\n")
            break

        # Best candidate selection logic: Lower priority rank, then higher acc, then higher layer index
        best_block = sorted(
            feasible,
            key=lambda b: (per_cand_pri[b], -per_cand_acc[b], -b)
        )[0]
        best_acc = per_cand_acc[best_block]
        best_pri = per_cand_pri[best_block]

        removal_list.append(best_block)
        log_file.write(
            f"SELECT: layer={best_block:02d} | priority={best_pri} | acc={best_acc:.2f} | removal_list={removal_list}\n"
        )
        log_file.write("-" * 60 + "\n")
        log_file.flush()

    final_acc = eval_fn(removal_list.copy())
    log_file.write(f"[FINAL] {task_name} | removal_list={removal_list} | final_acc={final_acc:.2f}\n")
    log_file.write("=" * 80 + "\n\n")
    log_file.flush()

    return removal_list, final_acc

def main():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    os.makedirs("skip_logs", exist_ok=True)

    task_results = {}
    reference_lists_by_task = {}

    for removal_ratio in REMOVAL_RATIOS:
        master_log_path = os.path.join("skip_logs", f"qwen2_2b_freq_initial_multitask_linked_ratio_{removal_ratio:.2f}.log")
        with open(master_log_path, "w") as mlog:
            mlog.write(f"[MULTI-TASK] Linked Greedy Removal | ratio_limit={removal_ratio} | "
                       f"ACC_THRESHOLD_RATIO={ACC_THRESHOLD_RATIO}\n")
            mlog.write(f"TASK ORDER: {TASK_ORDER}\n")
            mlog.write("=" * 80 + "\n\n")
            mlog.flush()

            for idx, task_name in enumerate(TASK_ORDER, start=1):
                # Build reference lists based on specific task dependencies
                ref_lists = []
                if task_name == "car":
                    if "traffic light" in reference_lists_by_task:
                        ref_lists = [reference_lists_by_task["traffic light"]]
                elif task_name == "obstacle" or task_name == "person":
                    tmp = []
                    if "traffic light" in reference_lists_by_task:
                        tmp.append(reference_lists_by_task["traffic light"])
                    if "car" in reference_lists_by_task:
                        tmp.append(reference_lists_by_task["car"])
                    ref_lists = tmp
                elif task_name == "bicycle":
                    if "traffic light" in reference_lists_by_task:
                        ref_lists = [reference_lists_by_task["traffic light"]]

                log_path = os.path.join("skip_logs", f"qwen2_2b_freq_initial_{idx:02d}_{task_name.replace(' ', '_')}_ratio_{removal_ratio:.2f}.log")
                with open(log_path, "w") as logf:
                    logf.write(f"Greedy Linked Removal for [{task_name}] | ratio_limit={removal_ratio}\n")

                    if task_name == "traffic light":
                        baseline_acc = evaluate_traffic_light([], model, processor, DEVICE, MAX_NEW_TOKENS[task_name])
                    else:
                        baseline_acc = evaluate_coda_single_category(task_name, [], model, processor, DEVICE, MAX_NEW_TOKENS[task_name])
                    threshold = baseline_acc * ACC_THRESHOLD_RATIO

                    logf.write(f"Baseline Acc (no removal): {baseline_acc:.2f} | Threshold: {threshold:.2f}\n")
                    logf.write("=" * 60 + "\n")
                    logf.flush()

                    removal_list, final_acc = greedy_select_for_one_task(
                        task_name=task_name,
                        model=model,
                        processor=processor,
                        total_blocks=TOTAL_BLOCKS,
                        removal_ratio_limit=removal_ratio,
                        baseline_acc=baseline_acc,
                        acc_threshold_ratio=ACC_THRESHOLD_RATIO,
                        log_file=logf,
                        ref_lists=ref_lists
                    )

                    task_results[task_name] = {
                        "removal_list": removal_list,
                        "baseline_acc": baseline_acc,
                        "final_acc": final_acc,
                        "threshold": threshold
                    }
                    reference_lists_by_task[task_name] = removal_list

                    mlog.write(f"[TASK {idx}/{len(TASK_ORDER)}] {task_name}\n")
                    mlog.write(f"  Baseline: {baseline_acc:.2f} | Final: {final_acc:.2f} | Thres: {threshold:.2f}\n")
                    mlog.write(f"  Removal list: {removal_list}\n")
                    
                    # Calculate Jaccard similarity between completed tasks
                    finished = list(reference_lists_by_task.values())
                    if len(finished) > 1:
                        from itertools import combinations
                        def jaccard(a, b):
                            a, b = set(a), set(b)
                            return (len(a & b) / len(a | b)) if len(a | b) > 0 else 0.0
                        js = [jaccard(finished[i], finished[j]) for i, j in combinations(range(len(finished)), 2)]
                        if js:
                            mlog.write(f"  Avg Jaccard among finished tasks: {sum(js)/len(js):.3f}\n")
                    mlog.write("-" * 80 + "\n")
                    mlog.flush()

            summary_path = os.path.join("skip_logs", f"qwen2_2b_freq_initial_summary_linked_ratio_{removal_ratio:.2f}.json")
            with open(summary_path, "w") as sf:
                json.dump({
                    "ratio_limit": removal_ratio,
                    "acc_threshold_ratio": ACC_THRESHOLD_RATIO,
                    "task_order": TASK_ORDER,
                    "results": task_results,
                    "reference_lists_by_task": reference_lists_by_task
                }, sf, indent=2)
            mlog.write(f"\nSaved summary to: {summary_path}\n")

if __name__ == "__main__":
    main()