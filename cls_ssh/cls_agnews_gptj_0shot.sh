python run_classification.py \
--model="gptj" \
--dataset="agnews" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.5 \
--val_size=300 \
--val_seed=20230307