#!/bin/sh

sbatch ./scripts/sweep-bom.slrm
sbatch ./scripts/sweep-hgm.slrm
sbatch ./scripts/sweep-hgm-lossweights.slrm
sbatch ./scripts/sweep-hgm-noman1d.slrm
