python run_classification.py \
--model="gptj" \
--dataset="rte" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=277 \
--epochs=50 \
--lr=0.00095 \
--val_size=100 \
--val_seed=20230307

