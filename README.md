# detection_logits

#### 1. Installation
All the codes are tested in the following environment:
* python 3.11.0
* pytorch 2.0.1
* numpy 1.26.0
* scikit-learn 1.3.1
* transformers 4.34.0
* fschat 0.2.30
* easydict 1.10
* tqdm 4.66.1

Set up and activate virtual environment: `python -m venv env; source env/bin/activate`

Install requirements: `pip install -r requirements.txt`

#### 2. Run

Configurations are in [configs.py](https://github.com/WhoTHU/detection_logits/blob/36bb1dc74ef91a714f4a9057a69b9387c1697e78/configs.py)

Run the code by
```
python main.py

main.py [-h] --model_name {llama-2,llama-2-13b,llama-2-70b,llama-3,flan-t5-small,flan-t5-large,flan-t5-xl,flan-t5-xxl,mistral-7b,gpt2,tiny-llama,vicuna-7b,vicuna-13b}
        [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--l1_reg L1_REG] [--hf_token_path HF_TOKEN_PATH] [--hf_token HF_TOKEN]
        [--device_list DEVICE_LIST] [--regression_device REGRESSION_DEVICE]

Logistic Regression Training With Model Logits

positional arguments:
  {llama-2,llama-2-13b,llama-2-70b,llama-3,flan-t5-small,flan-t5-large,flan-t5-xl,flan-t5-xxl,mistral-7b,gpt2,tiny-llama,vicuna-7b,vicuna-13b}
                        Name of the model

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS, -E EPOCHS
                        Number of epochs
  --batch_size BATCH_SIZE, -B BATCH_SIZE
                        Batch size
  --learning_rate LEARNING_RATE, -L LEARNING_RATE
                        Learning rate
  --l1_reg L1_REG, -R L1_REG
                        L1 regularization
  --hf_token_path HF_TOKEN_PATH
                        Path to the HF token file
  --hf_token HF_TOKEN   HF token
  --device_list DEVICE_LIST, -D DEVICE_LIST
                        Comma delimited list of device indices
  --regression_device REGRESSION_DEVICE
                        Device for regression
  --data_set {toxic-chat,lmsys-chat-1m,lmsys-chat-1m-handlabeled-split}, -S {toxic-chat,lmsys-chat-1m,lmsys-chat-1m-handlabeled-split}
                        Name of dataset
```

To add a new model, create a new sub-class of `DetectionModelConfig` in `configs.py`, specifying local path and/or HuggingFace path. Add the new config to the `ALL_MODEL_CONFIGS` collection.

To choose a data set, see `dataset_configs` in `configs.py`.