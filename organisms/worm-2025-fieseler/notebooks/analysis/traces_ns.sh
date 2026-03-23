for i in $(seq 0 17); do
  papermill traces_model_ns.ipynb traces_model_ns_f$i.ipynb -p file_id $i
done
