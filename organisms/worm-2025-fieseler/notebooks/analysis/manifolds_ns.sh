for i in $(seq 0 17); do
  papermill manifolds_model_ns.ipynb manifolds_model_ns_f$i.ipynb -p file_id $i
done
