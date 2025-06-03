#!/usr/bin/env bash
# TODO: FIXME WIP
set -e

for f in 00_easy 01_hard 02_ultimate; do
   echo "Running $f...";
   jupyter nbconvert --to notebook --execute $f/*_suite.ipynb --inplace;
   python - <<'PY'
import json,sys,nbformat,os,subprocess as sp
nb=sys.argv[1]; log=nb.replace('_suite.ipynb','_log.txt')
sp.run(['jupyter','nbconvert','--to','script',nb,'--stdout'],stdout=open(log,'w'))
PY $f/*_suite.ipynb;
done