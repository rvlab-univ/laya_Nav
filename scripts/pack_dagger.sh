#!/usr/bin/env bash
# Pack a DAgger round collected on this PC for training on the server (scripts/server/experiment.sh --extra):
# its annotations and stored features, about 1 MB per rollout.
#   bash scripts/pack_dagger.sh 1          # -> dagger1.tar; on the server: tar -xf dagger1.tar -C ~/light_nav
set -eu
ROUND="${1:?round number}"
cd "$(dirname "$0")/.."
D="data/dagger/round$ROUND"
F="data/features/siglip2_g2/dagger$ROUND"
[ -f "$D/annotations.json" ] && [ -d "$F" ] || { echo "round $ROUND is not here: $D, $F"; exit 1; }
n="$(env/bin/python -c "import json,sys; print(len(json.load(open('$D/config.json'))['episodes']))")"
[ "$(ls "$F" | wc -l)" = "$n" ] || { echo "round $ROUND is not finished: $(ls "$F" | wc -l) of $n rollouts stored"; exit 1; }
tar -cf "dagger$ROUND.tar" "$D/annotations.json" "$D/config.json" "$D/summary.json" "$F"
ls -lh "dagger$ROUND.tar"
