#!/usr/bin/env bash
# What is in place on this machine: code, GPU, disk, environments, data, teacher checkpoint.
#   bash scripts/server/check.sh            (TESTS=1 also runs the student tests, about a minute)
source "$(dirname "$0")/common.sh"
set +e
ok()   { echo "  ok    $*"; }
miss() { echo "  MISS  $*"; }

echo "== code"
echo "  $(git -C "$ROOT" log --oneline -1)"
git -C "$ROOT" submodule status | sed 's/^/  /'

echo "== GPU and disk"
gpu_report | sed 's/^/  /'
df -h "$ROOT" "$DATA" 2>/dev/null | tail -n +2 | sort -u | sed 's/^/  /'

echo "== environments"
conda_hook
for e in "$STUDENT_ENV" "$TEACHER_ENV"; do has_env "$e" && ok "conda env $e" || miss "conda env $e"; done
if has_env "$STUDENT_ENV"; then
    conda activate "$STUDENT_ENV"
    python -c "import torch, transformers, laya; print('  student: torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
        || miss "student imports"
fi
if has_env "$TEACHER_ENV"; then
    conda activate "$TEACHER_ENV"
    python -c "import torch, transformers, flash_attn, llava.model.builder; print('  teacher: torch', torch.__version__, '| transformers', transformers.__version__)" \
        || miss "teacher imports"
fi

echo "== data ($NAVILA_DATA)"
for f in R2R/annotations.json RxR/annotations.json ScanQA/annotations/ScanQA_v1.0_train_reformat.json; do
    [ -f "$NAVILA_DATA/$f" ] && ok "$f" || miss "$f"
done
for d in R2R/train:10819 RxR/train:6660 ScanQA/videos:800; do
    p="${d%%:*}"; want="${d##*:}"
    if [ -d "$NAVILA_DATA/$p" ]; then
        n=$(ls "$NAVILA_DATA/$p" | wc -l); [ "$n" -ge "$want" ] && ok "$p: $n" || miss "$p: $n of $want"
    else
        miss "$p"
    fi
done
[ -f "$SPLIT" ] && ok "split $(basename "$SPLIT")" || miss "split $SPLIT"

echo "== teacher checkpoint ($TEACHER_CKPT)"
[ -f "$TEACHER_CKPT/config.json" ] && ok "config.json, $(du -sh "$TEACHER_CKPT" | cut -f1)" || miss "checkpoint"

echo "== runs"
ls "$RUNS" 2>/dev/null | sed 's/^/  /'

if [ "${TESTS:-0}" = 1 ] && has_env "$STUDENT_ENV"; then
    echo "== student tests"
    conda activate "$STUDENT_ENV"
    (cd "$ROOT" && python -m pytest tests -q -p no:cacheprovider)
fi
