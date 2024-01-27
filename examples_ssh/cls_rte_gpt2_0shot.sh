python run_classification.py \
--model="gpt2-xl" \
--dataset="rte" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=277 \
--epochs=50 \
--lr=0.95 \
--val_size=30 \
--val_seed=20230307
