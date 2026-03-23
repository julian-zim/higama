for i in $(seq 0 17); do
  papermill traces_model_ss.ipynb traces_model_ss_f$i.ipynb -p file_id $i
done
