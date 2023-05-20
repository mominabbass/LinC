python run_classification.py \
--model="gptj" \
--dataset="dbpedia" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=300 \
--epochs=50 \
--lr=1.35 \
--val_size=190 \
--val_seed=20230307