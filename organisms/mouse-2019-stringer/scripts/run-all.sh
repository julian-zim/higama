#!/bin/sh

for f in ./scripts/run-*.slrm; do sbatch "$f"; done
