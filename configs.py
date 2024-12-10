
hf_token = None # hugging face token if needed

device = 'cuda:1'

model_configs = {
    'target': {
        'name': 'llama-2',
        'path': 'meta-llama/Llama-2-7b-chat-hf', # lcoal path or hugging face path
    },
    # 'judgement': {
    #     'name': 'llamaguard',
    #     'path': 'meta-llama/LlamaGuard-7b',
    # },
}

dataset_configs = {
    # ########## toxic-chat
    # 'name': 'toxic-chat', # toxic-chat, lmsys-chat-1m
    # 'path': None, # or local path
    # 'logits_dir': './cache',

    ########## lmsys-chat-1m
    'name': 'lmsys-chat-1m', # toxic-chat, lmsys-chat-1m
    'path': "./data/handlabeled_toxicity_lmsys_1m_split_50.json", # local path
    'logits_dir': './cache',
}


