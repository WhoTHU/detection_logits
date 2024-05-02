import os

hf_token = None
if os.path.exists('.hf_token'):
    with open('.hf_token', 'r') as f:
        hf_token = f.read().strip()

device = 'cuda:1'

ALL_MODEL_CONFIGS = {
    'llama-2': {
        'name': 'llama-2',
        'path': '/data1/public_models/llama2/Llama-2-7b-chat-hf/', # local path
    },
    'llama-3': {
        'name': 'llama-3',
        'path': '/data1/public_models/llama3/Meta-Llama-3-8B-Instruct-HF', # HF path
    },
    'flan-t5-xl': { # TODO
        'name': 'flan-t5-xl',
        'path': 'google/flan-t5-xl', # HF path
    },
    'flan-t5-xxl': { # TODO
        'name': 'flan-t5-xxl',
        'path': 'google/flan-t5-xxl', # HF path
    },
    'mistral-7b': {
        'name': 'mistral-7b',
        'path': '/data1/public_models/mistral/Mistral-7B-Instruct-v0.2', # local path
    },
}

model_configs = {
    'target': ALL_MODEL_CONFIGS['mistral-7b']
    # 'judgement': {
    #     'name': 'llamaguard',
    #     'path': 'meta-llama/LlamaGuard-7b',
    # },
}

dataset_configs = {
    # ########## toxic-chat
    # 'name': 'toxic-chat', # toxic-chat, lmsys-chat-1m
    # 'path': None, # local path
    # 'logits_dir': './cache',

    ########## lmsys-chat-1m
    'name': 'lmsys-chat-1m', # toxic-chat, lmsys-chat-1m
    'path': None, # local path
    'select': 10000, # use the first several examples
    'logits_dir': './cache',
}


