## Usage

### Input data files
* lm_dataset.csv - CSV file, where in the first column are texts for pre-training a language model
* clas_dataset.csv - CSV file containing the classification task dataset,
where there are texts in the first column and labels in the second

### Language model pre-training
```
python prepare_lm_dataset.py lm_dataset.csv lm_dataset_dir
python train_lm.py lm_dataset_dir trained_lm_dir
python evaluate_lm.py lm_dataset.csv trained_lm_dir    # optional
```

### LM fine-tuning and classifier training 
```
python prepare_clas_dataset.py clas_dataset.csv clas_dataset_dir
python finetune_lm.py clas_dataset_dir trained_lm_dir finetuned_lm_dir
python train_clas.py clas_dataset_dir finetuned_lm_dir trained_clas_dir
```

### Preferred output directory structure
Use separate directories (directory names passed as arguments) to store processed LM dataset,
processed classification dataset (tokenized and filtered texts with vocabularies),
trained LM, finetuned LM and trained classifier (stored model weights, model encoder,
vocabulary and model hyper-parameters).

### Comparison to performance without pretrained / finetuned LM
To fine-tune langugage model without a pretrained LM first create an untrained LM using
`python train_lm.py lm_dataset_dir trained_lm_dir --epochs=0` and then continue with fine-tuning.

To train a classifier without a trained language model, use `train_clas.py` with parameter
`pretrained=0`.
