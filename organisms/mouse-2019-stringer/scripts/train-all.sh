#!/bin/sh

for f in ../../scripts/train-*.slrm; do sbatch "$f"; done
