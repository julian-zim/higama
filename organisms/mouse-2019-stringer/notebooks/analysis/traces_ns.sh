for i in $(seq 0 8); do
  papermill traces_model_ns.ipynb traces_model_ns_f$i.ipynb -p file_id $i -p save True

  rm traces_model_ns_f$i.ipynb
done
