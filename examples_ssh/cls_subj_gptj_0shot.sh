python run_classification.py \
--model="gptj" \
--dataset="subj" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=300 \
--epochs=15 \
--lr=0.15 \
--val_size=100 \
--val_seed=20230307
