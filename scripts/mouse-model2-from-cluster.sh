rsync -av \
  --prune-empty-dirs \
  --include='*/' \
  --include='spont_M160907_MP028_2016-09-26/***' \
  --exclude='*' \
  zimmermann@login01.lisc.univie.ac.at:/lisc/data/scratch/neurobiology/zimmer/zimmermann/MODEL/mouse-2019-stringer \
  /home/julian-zim/Desktop/MODEL/mouse-2019-stringer/