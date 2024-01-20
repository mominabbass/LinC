python run_classification.py \
--model="gptj" \
--dataset="trec" \
--num_seeds=5 \
--all_shots="1" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.1 \
--val_size=100 \
--val_seed=20230303