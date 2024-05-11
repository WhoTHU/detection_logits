import os
from abc import ABC, abstractmethod

class DetectionModelConfig(ABC):
    def __init__(self, name, hf_path: str = None, local_path: str = None,
                 epochs: int = 512, batch_size: int = 32, learning_rate: float = 5e-4,
                 l1_reg: float = 5e-4, max_length: int|None = None): # devices
        self.name = name
        self.hf_path = hf_path
        self.local_path = local_path
        self.path = local_path if local_path is not None and os.path.exists(local_path) else hf_path
        if self.path is None:
            raise ValueError('Either hf_path or local_path must be provided')
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.max_length = max_length
        # TODO: devices
    
    def __getitem__(self, key):
        return getattr(self, key)

class Llama2_7BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('llama-2-7b',
                         'meta-llama/Llama-2-7b-chat-hf',
                         '/data1/public_models/llama2/Llama-2-7b-chat-hf/',
                         **kwargs)
        
class Llama2_13BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('llama-2-13b',
                         'meta-llama/Llama-2-13b-chat-hf',
                        #  '/data1/public_models/llama3/Llama-2-13b-chat-hf',
                         **kwargs)
        
class Llama2_70BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('llama-2-70b',
                         'meta-llama/Llama-2-70b-chat-hf',
                         **kwargs)        

class Llama3_8BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('llama-3',
                         'meta-llama/Llama-3-8B-Instruct-HF',
                         '/data1/public_models/llama3/Meta-Llama-3-8B-Instruct-HF',
                         **kwargs)
        
class FlanT5SmallDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('flan-t5-small',
                         'google/flan-t5-small',
                         **kwargs)
        
class FlanT5LargeDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('flan-t5-large',
                         'google/flan-t5-large',
                         **kwargs)
        
class FlanT5XLDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('flan-t5-xl',
                         'google/flan-t5-xl',
                         '/data1/public_models/flan-t5/flan-t5-xl',
                         **kwargs)
        
class FlanT5XXLDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('flan-t5-xxl',
                         'google/flan-t5-xxl',
                         '/data1/public_models/flan-t5/flan-t5-xxl',
                         **kwargs)
        
class Mistral7BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('mistral-7b',
                         'meta-llama/Mistral-7B-Instruct-v0.2',
                         '/data1/public_models/mistral/Mistral-7B-Instruct-v0.2',
                         **kwargs)

class GPT2DetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('gpt2',
                         'openai-community/gpt2-large',
                         max_length=1024,
                         **kwargs)
        
class TinyLlamaDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('tiny-llama',
                         'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                         '/data1/public_models/llama/TinyLlama-1.1B-Chat-v1.0',
                         **kwargs)
        
class Vicuna7BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('vicuna-7b',
                         'lmsys/vicuna-7b-v1.5-16k',
                         '/data1/public_models/llama/vicuna-7b-v1.5-16k',
                         **kwargs)

class Vicuna13BDetectionConfig(DetectionModelConfig):
    def __init__(self, **kwargs):
        super().__init__('vicuna-13b',
                         'lmsys/vicuna-13b-v1.5-16k',
                         '/data1/public_models/llama/vicuna-13b-v1.5-16k',
                         **kwargs)

# Add more subclasses for other detection models

ALL_MODEL_CONFIGS = {
    'llama-2': Llama2_7BDetectionConfig(),
    'llama-2-13b': Llama2_13BDetectionConfig(),
    'llama-2-70b': Llama2_70BDetectionConfig(),
    'llama-3': Llama3_8BDetectionConfig(),
    'flan-t5-small': FlanT5SmallDetectionConfig(),
    'flan-t5-large': FlanT5LargeDetectionConfig(),
    'flan-t5-xl': FlanT5XLDetectionConfig(),
    'flan-t5-xxl': FlanT5XXLDetectionConfig(),
    'mistral-7b': Mistral7BDetectionConfig(),
    'gpt2': GPT2DetectionConfig(),
    'tiny-llama': TinyLlamaDetectionConfig(),
    'vicuna-7b': Vicuna7BDetectionConfig(),
    'vicuna-13b': Vicuna13BDetectionConfig(),
}

dataset_configs = {
    # ########## toxic-chat
    # 'name': 'toxic-chat', # toxic-chat, lmsys-chat-1m
    # 'path': '/data1/public_datasets/toxic-chat', # local path
    # 'logits_dir': './cache',

    ########## lmsys-chat-1m
    # 'name': 'lmsys-chat-1m', # toxic-chat, lmsys-chat-1m
    # 'path': None, # local path
    # 'select': 20000, # use the first several examples
    # 'logits_dir': './cache',

    ########## lmsys-chat-1m-handlabeled-split
    'name': 'lmsys-chat-1m-handlabeled-split', # toxic-chat, lmsys-chat-1m
    'path': '/home/geng/handlabeled_toxicity_lmsys_1m_split.json', # local path
    'logits_dir': './cache',

}

ALL_DATASET_CONFIGS = {
    'toxic-chat': {
        'name': 'toxic-chat', # toxic-chat, lmsys-chat-1m
        'path': '/data1/public_datasets/toxic-chat', # local path
        'logits_dir': './cache',
    },
    'lmsys-chat-1m': {
        'name': 'lmsys-chat-1m', # toxic-chat, lmsys-chat-1m
        'path': None, # local path
        'select': 20000, # use the first several examples
        'logits_dir': './cache',
    },
    'lmsys-chat-1m-handlabeled-split': {
        'name': 'lmsys-chat-1m-handlabeled-split', # toxic-chat, lmsys-chat-1m
        'path': '/data1/public_datasets/lmsys-chat-1m-hand-labeled/handlabeled_toxicity_lmsys_1m_split.json', # local path
        'logits_dir': './cache',
    }
}


