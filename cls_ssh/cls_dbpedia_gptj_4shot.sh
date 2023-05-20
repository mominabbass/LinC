python run_classification.py \
--model="gptj" \
--dataset="dbpedia" \
--num_seeds=5 \
--all_shots="4" \
--subsample_test_set=300 \
--epochs=100 \
--lr=0.0095 \
--val_size=30 \
--val_seed=20230307