python run_classification.py \
--model="gptj" \
--dataset="trec" \
--num_seeds=5 \
--all_shots="4" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.069 \
--val_size=100 \
--val_seed=20230302