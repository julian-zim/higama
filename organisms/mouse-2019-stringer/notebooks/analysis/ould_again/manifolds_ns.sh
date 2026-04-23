for i in $(seq 0 8); do
  papermill manifolds_model_ns.ipynb manifolds_model_ns_f$i.ipynb -p file_id $i -p save True

  rm manifolds_model_ns_f$i.ipynb
done
