# Linear Probe Calibration (LinC)

This codebase is compatible with GPT-2, GPT-J, Llama-2, and any other language model available in [HuggingFace Transformers](https://huggingface.co/models).


## Dependencies

The code is implemented using PyTorch and the [HuggingFace's Transformer repository](https://github.com/huggingface/pytorch-transformers). If you intend to run the code on a small local model like GPT-2, it necessitates a single GPU.

## Installation
To setup the anaconda environment, simply run the following command:
```
conda env create -f setup_environment.yaml
```

After installation is complete, run:
```
conda activate fewshot_a10
```

## Datasets
We provide evaluation support for SST-2, SST-5, AGNews, TREC, DBPedia, RTE, and Subj datasets. You have the flexibility to incorporate additional text-classification datasets by defining the prompt format and label space in a manner similar to the existing datasets in data_utils.py.

## Evaluation
You can replicate the results in our paper by running the example sh scripts in the `cls_ssh` folder. For example, to run SST-2 0-shot on GPT-J, run: `sh examples_ssh/cls_sst2_gptj_0shot.sh`. Alternatively, copy and paste the contents of the .sh file into the terminal as follows:

```
python run_classification.py \
--model="gptj" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="0" \
--subsample_test_set=300 \
--epochs=15 \
--lr=0.015 \
--val_size=100 \
--val_seed=20230307
```

To execute different experiments, specify the desired arguments in the above command from the corresponding .sh file. For any other expeirment, follow settings from the paper. 

## Citation
If you find our work, or this repository useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{abbas2024,
 author    = {Abbas, Momin and Zhou, Yi and Ram, Parikshit and Baracaldo, Nathalie and Samulowitz, Horst and Salonidis, Theodoros and Chen, Tianyi},
 title     = {Enhancing In-context Learning via Linear Probe Calibration},
 year      = {2024},
 booktitle = {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},,
 }

```

## Contact
Should you have any inquiries, reach out to us via email at abbasm2@rpi.edu.


## Acknowledgements

Our code is built upon [Contextual Calibration](https://github.com/tonyzhaozh/few-shot-learning) repository, and we extend our appreciation to the authors for sharing their code. Should you decide to use our model and code, we kindly request you to acknowledge and cite these works as well.

