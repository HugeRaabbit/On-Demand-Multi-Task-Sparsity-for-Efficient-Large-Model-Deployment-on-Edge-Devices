#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import torch
import gc
from copy import deepcopy

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = "path/to/your/llava-v1.6-vicuna-7b-hf"  
DEVICE = "cuda:0"

LOG_DIR = os.path.join(BASE_DIR, "logs")
LAYER_SHARD_DIR = os.path.join(BASE_DIR, "layer_shards")
EXPORT_LAYERS_ONCE = True

REMOVAL_SWITCH_CASES = [
    {
        "name": "SLEB: Car → Traffic Light",
        "removal_list_0": [13, 8, 10, 6, 2, 5, 12, 20, 16, 19, 25, 17, 22, 23],
        "removal_list_1": [29, 2, 9, 10, 16, 25, 19, 27, 21, 22],
    },
    {
        "name": "SLEB: Traffic Light → Car",
        "removal_list_0": [29, 2, 9, 10, 16, 25, 19, 27, 21, 22],
        "removal_list_1": [13, 8, 10, 6, 2, 5, 12, 20, 16, 19, 25, 17, 22, 23],
    },
    {
        "name": "SLEB: Car → Obstacle",
        "removal_list_0": [13, 8, 10, 6, 2, 5, 12, 20, 16, 19, 25, 17, 22, 23],
        "removal_list_1": [16, 11, 21, 22, 23, 15, 17, 19, 31, 27, 30, 7],
    },
    {
        "name": "SLEB: Obstacle → Person",
        "removal_list_0": [16, 11, 21, 22, 23, 15, 17, 19, 31, 27, 30, 7],
        "removal_list_1": [20, 18, 19, 4, 6, 7, 2, 17, 22, 23, 13],
    },
    {
        "name": "SLEB: Obstacle → Car",
        "removal_list_0": [16, 11, 21, 22, 23, 15, 17, 19, 31, 27, 30, 7],
        "removal_list_1": [13, 8, 10, 6, 2, 5, 12, 20, 16, 19, 25, 17, 22, 23],
    },
    {
        "name": "SLEB: Person → Obstacle",
        "removal_list_0": [20, 18, 19, 4, 6, 7, 2, 17, 22, 23, 13],
        "removal_list_1": [16, 11, 21, 22, 23, 15, 17, 19, 31, 27, 30, 7],
    },
    {
        "name": "Ours: Car → Traffic Light",
        "removal_list_0": [13, 14, 29, 26, 27, 25, 21, 17, 12, 20, 11, 2, 9, 28, 22],
        "removal_list_1": [29, 2, 27, 13, 12, 14, 25, 16, 26, 21, 20, 11, 17, 9, 31],
    },
    {
        "name": "Ours: Traffic Light → Car",
        "removal_list_0": [29, 2, 27, 13, 12, 14, 25, 16, 26, 21, 20, 11, 17, 9, 31],
        "removal_list_1": [13, 14, 29, 26, 27, 25, 21, 17, 12, 20, 11, 2, 9, 28, 22],
    },
    {
        "name": "Ours: Car → Obstacle",
        "removal_list_0": [13, 14, 29, 26, 27, 25, 21, 17, 12, 20, 11, 2, 9, 28, 22],
        "removal_list_1": [16, 11, 21, 25, 17, 20, 12, 27, 31, 9, 28, 22, 7, 29, 30, 2],
    },
    {
        "name": "Ours: Obstacle → Person",
        "removal_list_0": [16, 11, 21, 25, 17, 20, 12, 27, 31, 9, 28, 22, 7, 29, 30, 2],
        "removal_list_1": [20, 29, 16, 12, 2, 9, 27, 25, 17, 31, 26, 10, 23, 22, 24, 15],
    },
    {
        "name": "Ours: Obstacle → Car",
        "removal_list_0": [16, 11, 21, 25, 17, 20, 12, 27, 31, 9, 28, 22, 7, 29, 30, 2],
        "removal_list_1": [13, 14, 29, 26, 27, 25, 21, 17, 12, 20, 11, 2, 9, 28, 22],
    },
    {
        "name": "Ours: Person → Obstacle",
        "removal_list_0": [20, 29, 16, 12, 2, 9, 27, 25, 17, 31, 26, 10, 23, 22, 24, 15],
        "removal_list_1": [16, 11, 21, 25, 17, 20, 12, 27, 31, 9, 28, 22, 7, 29, 30, 2],
    },
]


# ============================
# Utils: mapping / safety
# ============================
def _ensure_orig_layer_ids(module):
    """
    Ensure `module._orig_layer_ids` exists to track indices
    relative to the original full model.
    """
    if not hasattr(module, "_orig_layer_ids") or module._orig_layer_ids is None:
        L = len(module.layers)
        module._orig_layer_ids = list(range(L))


# ============================
# Disk Shard Store
# ============================
class LayerDiskStore:
    """
    Handles per-layer weight storage and retrieval from disk.
    """

    def __init__(self, base_model, save_dir):
        self.base_model = base_model
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def export_layers(self):
        layers = self.base_model.language_model.model.layers
        for i, layer in enumerate(layers):
            sd = layer.state_dict()
            path = os.path.join(self.save_dir, f"layer_{i}.pt")
            torch.save(sd, path)
        meta = {
            "num_layers": len(layers),
            "dtype": str(next(self.base_model.parameters()).dtype),
        }
        with open(os.path.join(self.save_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"[DiskStore] Exported {len(layers)} layers to {self.save_dir}")

    @torch.no_grad()
    def load_layer(self, oid: int, device: str = "cpu"):
        """
        Loads state_dict for a specific layer index from disk into a module prototype.
        Returns: (layer_module, io_seconds)
        """
        proto = deepcopy(self.base_model.language_model.model.layers[oid])
        proto.to("cpu")
        for p in proto.parameters():
            p.requires_grad_(False)

        path = os.path.join(self.save_dir, f"layer_{oid}.pt")
        t0 = time.time()
        sd = torch.load(path, map_location="cpu")
        io_sec = time.time() - t0

        proto.load_state_dict(sd, strict=True)
        proto = proto.to(device)
        return proto, io_sec


# ============================
# Build sub-model (keep-list)
# ============================
def build_submodel_with_layers(model, keep_list, preserve_order=True):
    """
    Builds a sub-network containing only the layers specified in keep_list.
    """
    assert hasattr(model, "language_model") and hasattr(
        model.language_model.model, "layers"
    ), "Unexpected model structure for LlavaNext"

    kept = keep_list[:]
    if not preserve_order:
        kept.sort()

    sub = deepcopy(model) 
    full_layers = list(sub.language_model.model.layers)
    new_layers = torch.nn.ModuleList([full_layers[i] for i in kept])

    sub.language_model.model.layers = new_layers
    sub.language_model.model._orig_layer_ids = kept[:]

    for i, layer in enumerate(sub.language_model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = i

    L = len(new_layers)
    if hasattr(sub.language_model.model, "config"):
        sub.language_model.model.config.num_hidden_layers = L
    if hasattr(sub.language_model, "config"):
        sub.language_model.config.num_hidden_layers = L
    if hasattr(sub, "config"):
        sub.config.num_hidden_layers = L

    return sub


# ============================
# Incremental switch (base_model→GPU)
# ============================
@torch.no_grad()
def incremental_switch_layers(
    base_model,
    eval_model,
    target_keep_list,
    device=DEVICE,
    preserve_order=False,
    log_fn=None,
):
    """
    Switches model layers incrementally by deleting unused layers and adding 
    required layers from the base model directly to the GPU.
    """
    lm = eval_model.language_model.model
    _ensure_orig_layer_ids(lm)

    tgt = target_keep_list[:]
    if not preserve_order:
        tgt.sort()

    cur_orig_ids = lm._orig_layer_ids
    cur_set = set(cur_orig_ids)
    tgt_set = set(tgt)

    to_remove = list(cur_set - tgt_set)
    to_add = list(tgt_set - cur_set)

    if log_fn:
        log_fn(f"[INC] current keep: {cur_orig_ids}")
        log_fn(f"[INC] target  keep: {tgt}")
        log_fn(f"[INC] to_remove: {sorted(to_remove)}")
        log_fn(f"[INC] to_add   : {sorted(to_add)}")

    pos_map = {orig_id: idx for idx, orig_id in enumerate(lm._orig_layer_ids)}
    cur_pos_to_delete = sorted([pos_map[k] for k in to_remove if k in pos_map])

    for pos in reversed(cur_pos_to_delete):
        if log_fn:
            log_fn(f"[DEL] pos={pos} (orig={lm._orig_layer_ids[pos]})")
        del lm.layers[pos]
        del lm._orig_layer_ids[pos]

    tgt_pos_map = {oid: i for i, oid in enumerate(tgt)}
    sortable_pairs = [
        (tgt_pos_map.get(oid, 10**9), idx)
        for idx, oid in enumerate(lm._orig_layer_ids)
    ]
    new_order = [p[1] for p in sorted(sortable_pairs, key=lambda x: (x[0], x[1]))]
    lm.layers = torch.nn.ModuleList([lm.layers[i] for i in new_order])
    lm._orig_layer_ids = [lm._orig_layer_ids[i] for i in new_order]

    h2d_records = []
    h2d_sum = 0.0
    for oid in sorted(to_add, key=lambda x: tgt_pos_map[x]):
        idx_tgt = tgt_pos_map[oid]
        insert_pos = None
        for cur_idx, exist_oid in enumerate(lm._orig_layer_ids):
            if tgt_pos_map[exist_oid] >= idx_tgt:
                insert_pos = cur_idx
                break
        if insert_pos is None:
            insert_pos = len(lm._orig_layer_ids)

        base_layer = base_model.language_model.model.layers[oid]
        new_layer = deepcopy(base_layer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        new_layer = new_layer.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0

        h2d_records.append((oid, dt))
        h2d_sum += dt

        lm.layers.insert(insert_pos, new_layer)
        lm._orig_layer_ids.insert(insert_pos, oid)
        if log_fn:
            log_fn(f"[INS] orig={oid} → pos={insert_pos}, H2D={dt:.6f}s")

    final_pos_map = {oid: idx for idx, oid in enumerate(lm._orig_layer_ids)}
    lm.layers = torch.nn.ModuleList([lm.layers[final_pos_map[oid]] for oid in tgt])
    lm._orig_layer_ids = tgt[:]

    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = i
    L = len(lm.layers)
    if hasattr(lm, "config"):
        lm.config.num_hidden_layers = L
    if hasattr(eval_model.language_model, "config"):
        eval_model.language_model.config.num_hidden_layers = L
    if hasattr(eval_model, "config"):
        eval_model.config.num_hidden_layers = L

    if log_fn:
        log_fn(f"[DONE] switched to: {lm._orig_layer_ids} (L={L})")
        log_fn(f"[H2D] incremental add total: {h2d_sum:.6f}s")

    return {"h2d_seconds_sum": h2d_sum, "per_layer": h2d_records}


def removal_to_keep(removal_list, num_layers):
    all_ids = set(range(num_layers))
    keep = sorted(list(all_ids - set(removal_list)))
    return keep

def profile_single_layers_and_full_model(base_model, layer_store, device, log_file):
    dev = torch.device(device)
    num_layers = len(base_model.language_model.model.layers)

    single_layer_records = []

    print("\n[Part 1] Profiling single layers (SSD->CPU & CPU->GPU & GPU memory) ...")
    log_file.write("\n[Part 1] Single layer profiling\n")
    log_file.flush()

    for oid in range(num_layers):
        lyr_cpu, io_sec = layer_store.load_layer(oid, device="cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        lyr_gpu = lyr_cpu.to(dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        h2d_sec = time.time() - t0

        if torch.cuda.is_available():
            gpu_mem_bytes = torch.cuda.max_memory_allocated(dev)
        else:
            gpu_mem_bytes = 0
        gpu_mem_mib = gpu_mem_bytes / (1024.0**2)

        rec = {
            "layer_id": oid,
            "io_ms": io_sec * 1000.0,
            "h2d_ms": h2d_sec * 1000.0,
            "gpu_mem_mib": gpu_mem_mib,
        }
        single_layer_records.append(rec)

        log_file.write(
            f"  [Layer {oid:02d}] IO={rec['io_ms']:.3f} ms, "
            f"H2D={rec['h2d_ms']:.3f} ms, "
            f"GPU mem≈{rec['gpu_mem_mib']:.2f} MiB\n"
        )
        log_file.flush()

        del lyr_cpu, lyr_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    avg_io = sum(r["io_ms"] for r in single_layer_records) / len(single_layer_records)
    avg_h2d = sum(r["h2d_ms"] for r in single_layer_records) / len(single_layer_records)
    avg_mem = sum(r["gpu_mem_mib"] for r in single_layer_records) / len(single_layer_records)

    log_file.write(
        f"\n[Single Layer Summary] avg IO={avg_io:.3f} ms, "
        f"avg H2D={avg_h2d:.3f} ms, avg GPU mem≈{avg_mem:.2f} MiB\n"
    )
    log_file.flush()
    print(
        f"[Single Layer Summary] avg IO={avg_io:.3f} ms, "
        f"avg H2D={avg_h2d:.3f} ms, avg GPU mem≈{avg_mem:.2f} MiB"
    )

    print("\n[Part 1] Profiling full model (CPU->GPU & GPU memory) ...")
    log_file.write("\n[Part 1] Full model profiling\n")
    log_file.flush()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    full_model = deepcopy(base_model)
    t0 = time.time()
    full_model = full_model.to(dev)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    full_h2d_sec = time.time() - t0

    if torch.cuda.is_available():
        full_gpu_bytes = torch.cuda.max_memory_allocated(dev)
    else:
        full_gpu_bytes = 0
    full_gpu_mib = full_gpu_bytes / (1024.0**2)

    log_file.write(
        f"  [Full Model] H2D={full_h2d_sec*1000.0:.3f} ms, "
        f"GPU mem≈{full_gpu_mib:.2f} MiB\n"
    )
    log_file.flush()
    print(
        f"[Full Model] H2D={full_h2d_sec*1000.0:.3f} ms, "
        f"GPU mem≈{full_gpu_mib:.2f} MiB"
    )

    del full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "single_layers": single_layer_records,
        "single_layer_avg": {
            "io_ms": avg_io,
            "h2d_ms": avg_h2d,
            "gpu_mem_mib": avg_mem,
        },
        "full_model": {
            "h2d_ms": full_h2d_sec * 1000.0,
            "gpu_mem_mib": full_gpu_mib,
        },
    }

def profile_removal_switch_cases(base_model, device, log_file):
    """
    Profiles switching between sub-networks defined by removal lists.
    """
    dev = torch.device(device)
    num_layers = len(base_model.language_model.model.layers)

    results = []

    log_file.write("\n[Part 2] Removal-list switch profiling\n")
    log_file.flush()
    print("\n[Part 2] Profiling removal_list switch cases ...")

    for case in REMOVAL_SWITCH_CASES:
        name = case.get("name", "unnamed_case")
        r0 = case["removal_list_0"]
        r1 = case["removal_list_1"]

        keep0 = removal_to_keep(r0, num_layers)
        keep1 = removal_to_keep(r1, num_layers)

        log_file.write(
            f"\n[Case: {name}] removal_list_0={r0}, removal_list_1={r1}\n"
        )
        log_file.write(f"  keep_list_0={keep0}\n")
        log_file.write(f"  keep_list_1={keep1}\n")
        log_file.flush()

        eval_model = build_submodel_with_layers(base_model, keep0, preserve_order=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize()
        t0 = time.time()
        eval_model = eval_model.to(dev).eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        h2d0_sec = time.time() - t0

        if torch.cuda.is_available():
            gpu0_bytes = torch.cuda.max_memory_allocated(dev)
        else:
            gpu0_bytes = 0
        gpu0_mib = gpu0_bytes / (1024.0**2)

        log_file.write(
            f"  [Case {name}] removal_list_0 submodel: H2D={h2d0_sec*1000.0:.3f} ms, "
            f"GPU mem≈{gpu0_mib:.2f} MiB\n"
        )
        log_file.flush()
        print(
            f"[Case {name}] removal_list_0 -> H2D={h2d0_sec*1000.0:.3f} ms, "
            f"GPU mem≈{gpu0_mib:.2f} MiB"
        )

        def _log(msg):
            log_file.write("    " + msg + "\n")
            log_file.flush()

        info = incremental_switch_layers(
            base_model,
            eval_model,
            target_keep_list=keep1,
            device=device,
            preserve_order=False,
            log_fn=_log,
        )

        h2d_switch_ms = info["h2d_seconds_sum"] * 1000.0

        log_file.write(
            f"  [Case {name}] incremental switch H2D_total={h2d_switch_ms:.3f} ms\n"
        )
        log_file.flush()
        print(
            f"[Case {name}] incremental switch H2D_total={h2d_switch_ms:.3f} ms"
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu1_bytes = torch.cuda.memory_allocated(dev)
        else:
            gpu1_bytes = 0
        gpu1_mib = gpu1_bytes / (1024.0**2)

        log_file.write(
            f"  [Case {name}] after switch, GPU mem≈{gpu1_mib:.2f} MiB\n"
        )
        log_file.flush()

        results.append(
            {
                "name": name,
                "removal_list_0": r0,
                "removal_list_1": r1,
                "keep_list_0": keep0,
                "keep_list_1": keep1,
                "removal0_h2d_ms": h2d0_sec * 1000.0,
                "removal0_gpu_mem_mib": gpu0_mib,
                "switch_h2d_ms": h2d_switch_ms,
                "after_switch_gpu_mem_mib": gpu1_mib,
            }
        )

        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "llava_7b_h2d_io_profile.log")
    summary_json_path = os.path.join(LOG_DIR, "llava_7b_h2d_io_profile_summary.json")

    with open(log_path, "w") as mlog:
        mlog.write("LLaVA H2D / IO / GPU Memory Profiling\n")
        mlog.write("=" * 80 + "\n\n")
        mlog.flush()

        print("[Init] Loading processor & base model (CPU only) ...")
        processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )

        layer_store = LayerDiskStore(base_model, LAYER_SHARD_DIR)
        if EXPORT_LAYERS_ONCE:
            layer_store.export_layers()
        else:
            print("[Note] EXPORT_LAYERS_ONCE=False, assuming layer shards exist.")

        part1_summary = profile_single_layers_and_full_model(
            base_model, layer_store, DEVICE, mlog
        )

        part2_summary = profile_removal_switch_cases(base_model, DEVICE, mlog)

        all_summary = {
            "device": DEVICE,
            "num_layers": len(base_model.language_model.model.layers),
            "part1_single_and_full": part1_summary,
            "part2_switch_cases": part2_summary,
        }
        with open(summary_json_path, "w") as sf:
            json.dump(all_summary, sf, indent=2)

        print(f"\n[Done] Summary json saved to: {summary_json_path}")
        print(f"[Done] Log saved to: {log_path}")


if __name__ == "__main__":
    main()