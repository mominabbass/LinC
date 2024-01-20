python run_classification.py \
--model="llama2_13b" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="1" \
--subsample_test_set=300 \
--epochs=1 \
--lr=0.000085 \
--val_size=300 \
--val_seed=20230303
