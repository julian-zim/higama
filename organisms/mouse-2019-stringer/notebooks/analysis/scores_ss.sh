for i in $(seq 0 17); do
  papermill scores_model_ss.ipynb scores_model_ss_f$i.ipynb -p file_id $i
done
