#!/usr/bin/env bash
# Pack a DAgger round collected on this PC for training on the server (scripts/server/experiment.sh --extra):
# its annotations, rollouts (episodes.jsonl: the actions that ran, for --pose) and stored features, about 1 MB per
# rollout.
#   bash scripts/pack_dagger.sh 1          # -> dagger1.tar and dagger1.tar.sha256
# When the round's frames were kept (scripts/dagger_round.sh, from round 2 on), they go into a second file,
# dagger<n>_frames.tar (+ .sha256, about 10 GB), for NaVILA's labels on the round's states: upload it too, then
# scripts/server/teacher_dagger.sh <n> on the server. Training does not need it.
# Upload both to ~/light_nav on the server: scripts/server/l_runs.sh checks the checksum and extracts the round
# (by hand: sha256sum -c dagger1.tar.sha256, then tar -xf dagger1.tar -C ~/light_nav).
set -eu
ROUND="${1:?round number}"
cd "$(dirname "$0")/.."
D="data/dagger/round$ROUND"
F="data/features/siglip2_g2/dagger$ROUND"
[ -f "$D/annotations.json" ] && [ -d "$F" ] || { echo "round $ROUND is not here: $D, $F"; exit 1; }
n="$(env/bin/python -c "import json,sys; print(len(json.load(open('$D/config.json'))['episodes']))")"
[ "$(ls "$F" | wc -l)" = "$n" ] || { echo "round $ROUND is not finished: $(ls "$F" | wc -l) of $n rollouts stored"; exit 1; }
tar -cf "dagger$ROUND.tar" "$D/annotations.json" "$D/config.json" "$D/summary.json" "$D/episodes.jsonl" "$F"
sha256sum "dagger$ROUND.tar" > "dagger$ROUND.tar.sha256"
ls -lh "dagger$ROUND.tar"
cat "dagger$ROUND.tar.sha256"
have="$( [ -d "$D/frames" ] && find "$D/frames" -mindepth 1 -maxdepth 1 -type d | wc -l || echo 0)"
if [ "$have" = "$n" ]; then
    tar -cf "dagger${ROUND}_frames.tar" "$D/frames"
    sha256sum "dagger${ROUND}_frames.tar" > "dagger${ROUND}_frames.tar.sha256"
    ls -lh "dagger${ROUND}_frames.tar"
elif [ "$have" = 0 ]; then
    echo "no frames kept for round $ROUND: no teacher labels for its states (training is not affected)"
else
    echo "frames of $have of $n rollouts in $D/frames: dagger${ROUND}_frames.tar not made"
fi
