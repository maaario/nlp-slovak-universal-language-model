## Usage
* lm_dataset.csv - csv file, where in the first column are texts
* clas_dataset.csv - csv file, where there are texts in the first column and labels in the second

### Language model pre-training
```
python prepare_lm_dataset lm_dataset.csv lm_dataset_dir
python train_lm lm_dataset_dir trained_lm_dir
python evaluate_lm lm_dataset.csv trained_lm_dir    # optional
```

### LM fine-tuning and classifier training 
```
python prepare_clas_dataset.py clas_dataset.csv clas_dataset_dir
python finetune_lm.py clas_dataset_dir trained_lm_dir finetuned_lm_dir
python train_clas.py clas_dataset_dir finetuned_lm_dir trained_clas_dir
```