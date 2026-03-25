for i in $(seq 2 2); do
  papermill traces_model_ns.ipynb traces_model_ns_f$i.ipynb -p file_id $i
done
