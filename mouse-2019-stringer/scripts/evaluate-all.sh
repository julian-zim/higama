#!/bin/sh

for f in ./scripts/evaluate-*.slrm; do sbatch "$f"; done
