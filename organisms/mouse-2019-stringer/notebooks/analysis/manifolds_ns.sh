for i in $(seq 2 2); do
  papermill manifolds_model_ns.ipynb manifolds_model_ns_f$i.ipynb -p file_id $i
done
