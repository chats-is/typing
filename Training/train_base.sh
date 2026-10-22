#!/bin/zsh
# The shipped model's full recipe, three stages from scratch on one prepared directory
# (Training/prepare_data.py output, cleaned): about 36 hours on the Mac mini (M4).
#
#   Training/train_base.sh Data/prepared-clean Data/run1
#
# Stage a: 80,000 steps from scratch, lm2's mix (dialogue .32, mixed .23, forum .16, news .13,
#          novels .05, TouchPal .05, People's Daily .03, lexicon words .03, English .01).
# Stage b: +20,000 steps at lr 2e-4, lm3's mix (adds web, techdocs, tatoeba).
# Stage c: +6,000 steps at lr 5e-5, the mix in the prepared directory's own sources.json (lm9's).
# The stage directories link to the prepared token files and carry only their sources.json.
# Each stage resumes from its checkpoint when run again. Run under caffeinate.
set -e
DATA=${1:?usage: train_base.sh <prepared dir> <out dir>}
OUT=${2:?usage: train_base.sh <prepared dir> <out dir>}
PY=${PY:-Data/venv-train/bin/python}
DATA=${DATA:A}
mkdir -p $OUT/a $OUT/b
for stage in a b; do
    for f in $DATA/*.bin $DATA/vocab.json; do ln -sf $f $OUT/$stage/; done
done
cat > $OUT/a/sources.json <<'EOF'
{"dialogue":0.32,"mixed":0.23,"forum":0.16,"news":0.13,"novels":0.05,"touchpal":0.05,"peoples_daily":0.03,"words":0.03,"english":0.01}
EOF
cat > $OUT/b/sources.json <<'EOF'
{"dialogue":0.28,"mixed":0.18,"news":0.09,"forum":0.12,"novels":0.04,"touchpal":0.05,"peoples_daily":0.02,"web":0.14,"techdocs":0.02,"tatoeba":0.02,"words":0.03,"english":0.01}
EOF
$PY Training/train.py --data $OUT/a --out $OUT/a.pt --steps 80000 >> $OUT/a.log 2>&1
$PY Training/train.py --data $OUT/b --out $OUT/b.pt --init $OUT/a.pt --lr 2e-4 --steps 20000 >> $OUT/b.log 2>&1
$PY Training/train.py --data $DATA --out $OUT/model.pt --init $OUT/b.pt --lr 5e-5 --steps 6000 >> $OUT/c.log 2>&1
grep "finished" $OUT/a.log $OUT/b.log $OUT/c.log
