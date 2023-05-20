python run_classification.py \
--model="gptj" \
--dataset="agnews" \
--num_seeds=5 \
--all_shots="8" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.00045 \
--val_size=100 \
--val_seed=202303014