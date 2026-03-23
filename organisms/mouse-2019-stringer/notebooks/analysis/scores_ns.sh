for i in $(seq 0 17); do
  papermill scores_model_ns.ipynb scores_model_ns_f$i.ipynb -p file_id $i
done
