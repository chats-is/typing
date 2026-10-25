#!/bin/zsh
# The shipped model's full recipe, three stages from scratch on one prepared directory
# (Training/prepare_data.py output, cleaned).
#
#   Training/train_base.sh Data/prepared-clean Data/run1
#
# Stage a: 80,000 steps from scratch (dialogue .32, mixed .23, forum .16, news .13, novels .05,
#          TouchPal .05, People's Daily .03, lexicon words .03, English .01).
# Stage b: +20,000 steps at lr 2e-4 (adds web, techdocs, tatoeba).
# Stage c: +6,000 steps at lr 5e-5 (the final mix).
# The stage directories link to the prepared token files and carry only their sources.json.
# Each stage resumes from its checkpoint when run again. Refuses data that does not match the
# SHA256SUMS written where it was prepared.
set -e
DATA=${1:?usage: train_base.sh <prepared dir> <out dir>}
OUT=${2:?usage: train_base.sh <prepared dir> <out dir>}
PY=${PY:-Data/venv-train/bin/python}
DATA=${DATA:A}
# The data must be exactly what was prepared: SHA256SUMS is written next to it on the machine that
# prepared it (docs/development.md), and every file it names must match here before anything trains.
if [[ ! -f $DATA/SHA256SUMS ]]; then
    echo "no $DATA/SHA256SUMS: write it where the data was prepared, then copy it with the data" >&2
    exit 1
fi
if ! (cd $DATA && shasum -a 256 -c SHA256SUMS --quiet); then
    echo "$DATA does not match its SHA256SUMS: copy the data again" >&2
    exit 1
fi
echo "data checked: $(wc -l < $DATA/SHA256SUMS | tr -d ' ') files match SHA256SUMS"
mkdir -p $OUT/a $OUT/b $OUT/c
for stage in a b c; do
    for f in $DATA/*.bin $DATA/vocab.json; do ln -sf $f $OUT/$stage/; done
done
cat > $OUT/a/sources.json <<'EOF'
{"dialogue":0.32,"mixed":0.23,"forum":0.16,"news":0.13,"novels":0.05,"touchpal":0.05,"peoples_daily":0.03,"words":0.03,"english":0.01}
EOF
cat > $OUT/b/sources.json <<'EOF'
{"dialogue":0.28,"mixed":0.18,"news":0.09,"forum":0.12,"novels":0.04,"touchpal":0.05,"peoples_daily":0.02,"web":0.14,"techdocs":0.02,"tatoeba":0.02,"words":0.03,"english":0.01}
EOF
cat > $OUT/c/sources.json <<'EOF'
{"dialogue":0.32,"mixed":0.14,"news":0.07,"forum":0.1,"novels":0.04,"touchpal":0.05,"peoples_daily":0.02,"web":0.09,"techdocs":0.02,"tatoeba":0.02,"words":0.08,"english":0.05}
EOF
$PY Training/train.py --data $OUT/a --out $OUT/a.pt --steps 80000 >> $OUT/a.log 2>&1
$PY Training/train.py --data $OUT/b --out $OUT/b.pt --init $OUT/a.pt --lr 2e-4 --steps 20000 >> $OUT/b.log 2>&1
$PY Training/train.py --data $OUT/c --out $OUT/model.pt --init $OUT/b.pt --lr 5e-5 --steps 6000 >> $OUT/c.log 2>&1
grep "finished" $OUT/a.log $OUT/b.log $OUT/c.log
