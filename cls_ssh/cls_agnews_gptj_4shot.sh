python run_classification.py \
--model="gptj" \
--dataset="agnews" \
--num_seeds=5 \
--all_shots="4" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.000095 \
--val_size=100 \
--val_seed=20230305