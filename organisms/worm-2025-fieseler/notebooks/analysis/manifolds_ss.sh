for i in $(seq 0 17); do
  papermill manifolds_model_ss.ipynb manifolds_model_ss_f$i.ipynb -p file_id $i
done
