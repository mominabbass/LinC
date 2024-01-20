python run_classification.py \
--model="llama2_13b" \
--dataset="sst5" \
--num_seeds=5 \
--all_shots="8" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.00045 \
--val_size=100 \
--val_seed=20230303
