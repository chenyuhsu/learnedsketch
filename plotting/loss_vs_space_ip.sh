
tmin=1329
fname="20 th"

#tmin=1359
#fname="50 th"

model="ru64"

## count min

python3 plot_loss_vs_space.py \
    --algo          "Count-Min" \
    --count_min     ../param_results/count_min/cmin_ip_${tmin}.npz \
    --learned       ../param_results/cutoff_count_min_param/cmin_ip_${tmin}_${model}_test.npz \
    --perfect       ../param_results/cutoff_count_min_param_perfect/cmin_ip_${tmin}_pcut_test.npz \
    --lookup_table  ../param_results/lookup_table_count_min/cmin_ip_${tmin}_test.npz \
    --model_names   "Learned Count-Min (NNet)" \
    --title         "IP - @ ${fname} test minute - model ${model}" \
    --model_sizes   0.0031 \
    --lookup_size   0.0035 \
    --x_lim         0 2 \
    --y_lim         0 200

## count sketch

python3 plot_loss_vs_space.py \
    --algo          "Count-Sketch" \
    --count_min     ../param_results/count_sketch/csketch_ip_${tmin}.npz \
    --learned       ../param_results/cutoff_count_sketch_param/csketch_ip_${tmin}_${model}_test.npz \
    --perfect       ../param_results/cutoff_count_sketch_param_perfect/csketch_ip_${tmin}_pcut_test.npz  \
    --lookup_table  ../param_results/lookup_table_count_sketch/csketch_ip_${tmin}_test.npz \
    --model_names   "Learned Count-Sketch (NNet)" \
    --title         "IP - @ ${fname} test minute - model ${model}" \
    --model_sizes   0.0031 \
    --lookup_size   0.0035 \
    --x_lim         0 2 \
    --y_lim         0 200
