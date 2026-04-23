for i in $(seq 0 9); do
  papermill scores_model_ss.ipynb scores_model_ss_f$i.ipynb -p file_id $i -p save True

  rm scores_model_ss_f$i.ipynb
done
