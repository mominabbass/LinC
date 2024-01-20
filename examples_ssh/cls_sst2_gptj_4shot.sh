python run_classification.py \
--model="gptj" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="4" \
--subsample_test_set=300 \
--epochs=15 \
--lr=0.00035 \
--val_size=100 \
--val_seed=202303026