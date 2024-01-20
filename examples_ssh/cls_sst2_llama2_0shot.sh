python run_classification.py \
--model="llama2_13b" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=300 \
--epochs=15 \
--lr=0.0055 \
--val_size=100 \
--val_seed=20230307
