"""Teacher probabilities from NaVILA 8B over its 10 answers, for distillation (run N5).

Runs in the NaVILA environment (scripts/server/setup_teacher_env.sh), not the student one. For every unique R2R /
RxR decision sample it builds NaVILA's own prompt and 8 frames (black padding included, as NaVILA trains), reads the
prompt once and scores the 10 answer sentences on copies of its cache (lightnav/teacher.py).

Results go to <out>/shard_<k>/part_<n>.npz every --chunk samples. A rerun skips every key already in any shard of
<out>, so a stopped job resumes and shards can be added. --merge then writes keys.txt and teacher_probs.npy (what
lightnav.data.TeacherStore reads), teacher_logp.npy (raw log-likelihoods) and stats.json.

    # 1,000 samples, the first 20 also through model.generate as a check of the scoring
    python scripts/extract_teacher_probs.py --ckpt <ckpt> --out runs/teacher_check --limit 1000 --check_generate 20
    # one of 4 processes sharing a GPU
    python scripts/extract_teacher_probs.py --ckpt <ckpt> --out runs/teacher --num_shards 4 --shard 0
    python scripts/extract_teacher_probs.py --out runs/teacher --merge
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "NaVILA"))

from lightnav import data as D                                          # noqa: E402
from lightnav.questions import ANSWERS, KEYS, NUM_ACTIONS               # noqa: E402
from lightnav.teacher import answer_from_text, answer_token_ids, navila_question, score_answers  # noqa: E402

NOT_CHECKED = -2      # gen column: generate was not run for this sample
UNPARSED = -1         # generate ran, but NaVILA's evaluation loop would not read one of the 10 answers from it


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.path.join(ROOT, "data", "NaVILA-Dataset")
    p.add_argument("--ckpt", default=None, help="NaVILA checkpoint folder (a8cheng/navila-llama3-8b-8f)")
    p.add_argument("--out", required=True)
    p.add_argument("--sources", default="r2r,rxr")
    p.add_argument("--r2r_ann", default=os.path.join(data, "R2R", "annotations.json"))
    p.add_argument("--r2r_frames", default=os.path.join(data, "R2R", "train"))
    p.add_argument("--rxr_ann", default=os.path.join(data, "RxR", "annotations.json"))
    p.add_argument("--rxr_frames", default=os.path.join(data, "RxR", "train"))
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only the first N samples of this shard")
    p.add_argument("--chunk", type=int, default=1000, help="samples per saved part")
    p.add_argument("--check_generate", type=int, default=0,
                   help="also run model.generate on the first N samples and compare with the scores")
    p.add_argument("--merge", action="store_true", help="combine the parts in --out and write the outputs")
    p.add_argument("--skip_expected", action="store_true",
                   help="with --merge: do not count samples missing from the annotations (for --limit runs)")
    return p.parse_args()


def load_samples(args):
    """Unique decision samples (one per video_id and source), English only as the student uses them, by key."""
    rows, roots = [], {}
    for name in args.sources.split(","):
        rows += D.load_rows(getattr(args, name + "_ann"), name, english_only=True, dedupe=True)
        roots[name] = getattr(args, name + "_frames")
    return sorted(rows, key=lambda r: r.key), roots


def read_parts(out_dir):
    parts = sorted(glob.glob(os.path.join(out_dir, "shard_*", "part_*.npz")))
    return [np.load(p, allow_pickle=False) for p in parts if not p.endswith(".tmp.npz")]   # not one being written


def save_part(shard_dir, rec):
    os.makedirs(shard_dir, exist_ok=True)
    n = len(glob.glob(os.path.join(shard_dir, "part_*.npz")))
    path = os.path.join(shard_dir, "part_%05d.npz" % n)
    tmp = path + ".tmp.npz"
    np.savez(tmp, keys=np.array(rec["keys"]), labels=np.array(rec["labels"], dtype=np.int64),
             logp=np.array(rec["logp"], dtype=np.float32), gen=np.array(rec["gen"], dtype=np.int64))
    os.replace(tmp, path)
    for v in rec.values():
        v.clear()


def run(args):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token, vlnce_frame_sampling
    from llava.model.builder import load_pretrained_model

    rows, roots = load_samples(args)
    mine = rows[args.shard::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]
    done = {k for part in read_parts(args.out) for k in part["keys"].tolist()}
    todo = [r for r in mine if r.key not in done]
    print("shard %d/%d: %d samples, %d already done, %d to go"
          % (args.shard, args.num_shards, len(mine), len(mine) - len(todo), len(todo)), flush=True)
    if not todo:
        return

    ckpt = os.path.normpath(args.ckpt)
    tokenizer, model, image_processor, _ = load_pretrained_model(ckpt, os.path.basename(ckpt))
    model = model.cuda().eval()
    num_frames = model.config.num_video_frames
    answers = answer_token_ids(tokenizer)
    conv0 = conv_templates["llama_3"]
    stop_str = conv0.sep if conv0.sep_style != SeparatorStyle.TWO else conv0.sep2
    print("model loaded: %d frames, answer lengths %s tokens" % (num_frames, [len(a) for a in answers]), flush=True)

    shard_dir = os.path.join(args.out, "shard_%d" % args.shard)
    rec = {"keys": [], "labels": [], "logp": [], "gen": []}
    checks = {"run": 0, "agree": 0}
    missing, started = 0, time.time()
    for i, r in enumerate(todo):
        try:
            paths = [os.path.join(roots[r.source], r.frame(f)) for f in range(r.n_frames)]
            images = vlnce_frame_sampling(paths, num_frames)
        except (FileNotFoundError, OSError) as e:
            missing += 1
            print("missing frames for %s: %s" % (r.key, e), flush=True)
            continue
        conv = conv0.copy()
        conv.append_message(conv.roles[0], navila_question(D.clean_instruction(r.instruction), num_frames))
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        pixels = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        with torch.inference_mode():
            _, _, mask, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None,
                                                                                   pixels)
            logp = score_answers(model.llm, embeds.to(model.dtype), answers, mask).cpu().numpy()
            gen = NOT_CHECKED
            if checks["run"] < args.check_generate:
                out = model.generate(input_ids, images=pixels, do_sample=False, temperature=0.0, max_new_tokens=32,
                                     use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                     stopping_criteria=[KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)])
                text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
                if text.endswith(stop_str):
                    text = text[:-len(stop_str)].strip()
                parsed = answer_from_text(text)
                gen = UNPARSED if parsed is None else parsed
                checks["run"] += 1
                checks["agree"] += int(gen == int(logp.argmax()))
                print("check %s: generated %r -> %s | best score %s | label %s"
                      % (r.key, text, KEYS[gen] if gen >= 0 else "unparsed", KEYS[int(logp.argmax())],
                         KEYS[r.label]), flush=True)
        rec["keys"].append(r.key)
        rec["labels"].append(r.label)
        rec["logp"].append(logp)
        rec["gen"].append(gen)
        if len(rec["keys"]) >= args.chunk:
            save_part(shard_dir, rec)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            rate = (i + 1) / (time.time() - started)
            print("shard %d: %d/%d, %.2f samples/s, %.1f h left"
                  % (args.shard, i + 1, len(todo), rate, (len(todo) - i - 1) / rate / 3600), flush=True)
    if rec["keys"]:
        save_part(shard_dir, rec)
    if checks["run"]:
        print("generate check: %d/%d samples give the same answer as the best score"
              % (checks["agree"], checks["run"]), flush=True)
    if missing:
        print("samples skipped for missing frames: %d" % missing, flush=True)


def merge(args):
    parts = read_parts(args.out)
    if not parts:
        sys.exit("no parts under %s" % args.out)
    keys = np.concatenate([p["keys"] for p in parts])
    labels = np.concatenate([p["labels"] for p in parts])
    logp = np.concatenate([p["logp"] for p in parts])
    gen = np.concatenate([p["gen"] for p in parts])
    _, first = np.unique(keys, return_index=True)      # sorted by key, one row per key
    keys, labels, logp, gen = keys[first], labels[first], logp[first], gen[first]
    shifted = logp - logp.max(1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)

    with open(os.path.join(args.out, "keys.txt"), "w") as f:
        f.write("\n".join(keys.tolist()) + "\n")
    np.save(os.path.join(args.out, "teacher_probs.npy"), probs.astype(np.float32))
    np.save(os.path.join(args.out, "teacher_logp.npy"), logp.astype(np.float32))

    rows = np.arange(len(keys))
    top = probs.argmax(1)
    checked = gen != NOT_CHECKED
    stats = {
        "n": int(len(keys)),
        "by_source": {s: int(sum(k.startswith(s + ":") for k in keys.tolist())) for s in args.sources.split(",")},
        "top1_equals_label": float((top == labels).mean()),
        "mean_prob_of_label": float(probs[rows, labels].mean()),
        "mean_prob_off_label": float(1 - probs[rows, labels].mean()),
        "mean_max_prob": float(probs.max(1).mean()),
        "top1_equals_label_by_answer": {KEYS[i]: float((top[labels == i] == i).mean())
                                        for i in range(NUM_ACTIONS) if (labels == i).any()},
        "generate_checked": int(checked.sum()),
        "generate_equals_top1": float((gen[checked] == top[checked]).mean()) if checked.any() else None,
        "answers": ANSWERS,
    }
    if not args.skip_expected:
        expected = {r.key for r in load_samples(args)[0]}
        stats["expected"] = len(expected)
        stats["missing"] = len(expected - set(keys.tolist()))
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "answers"}, indent=2))


if __name__ == "__main__":
    a = parse_args()
    if a.merge:
        merge(a)
    else:
        if not a.ckpt:
            sys.exit("--ckpt is required unless --merge")
        run(a)
