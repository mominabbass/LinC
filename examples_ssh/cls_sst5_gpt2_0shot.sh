python run_classification.py \
--model="gpt2-xl" \
--dataset="sst5" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.0095 \
--val_size=30 \
--val_seed=20230307
