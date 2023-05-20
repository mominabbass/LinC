python run_classification.py \
--model="gptj" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="1" \
--subsample_test_set=300 \
--epochs=1 \
--lr=0.00015 \
--val_size=300 \
--val_seed=202303017

