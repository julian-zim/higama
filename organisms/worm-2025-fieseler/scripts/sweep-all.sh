#!/bin/sh

for f in ../scripts/sweep-*.slrm; do sbatch "$f"; done
