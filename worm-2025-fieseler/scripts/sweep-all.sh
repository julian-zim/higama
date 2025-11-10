#!/bin/sh

for f in ./scripts/sweep*; do sbatch "$f"; done
