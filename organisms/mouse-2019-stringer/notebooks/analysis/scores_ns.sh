for i in $(seq 0 9); do
  papermill scores_model_ns.ipynb scores_model_ns_f$i.ipynb -p file_id $i -p save True

  rm scores_model_ns_f$i.ipynb
done
