import os
import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
import numpy as np
import math
import json
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as roc_auc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from datasets import load_dataset
from easydict import EasyDict
from tqdm.auto import tqdm
from fastchat.model import get_conversation_template

def _load_distributed_hf(raw_config, devices, hf_token=None):
    model_name = raw_config.name
    model_path = raw_config.path
    model_dtype = torch.float32 if 'flan-t5' in model_path.lower() else torch.float16

    # Setup model config and memory map
    gpu_count = len(devices)
    if gpu_count <= 1:
        raise ValueError("Distributed inference requires at least 2 GPUs.")
    # if isinstance(devices[0], torch.device):
    #     gpu_indices = [d.index for d in devices]
    # else:
    #     gpu_indices = devices
    config = AutoConfig.from_pretrained(model_path, token=hf_token)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        total_size = model.num_parameters() * 2 / 10**9
        max_memory = {
            i: (
                "{}GiB".format(
                    math.ceil(1 / (gpu_count - 1) + total_size / gpu_count)
                )
                # if i in gpu_indices
                # else "0GiB"
            )
            for i in devices
        }
        max_memory[min(devices)] = "{}GiB".format(
            math.ceil(total_size / gpu_count - 1)
        )
        print(f"Memory allocation: {max_memory}")

        # Infer not split module:
        if "t5" in model_name:
            no_split_module_classes = ["T5Block"]
        elif (
            "vicuna" in model_name
            or "koala" in model_name
            or "llama" in model_name
        ):
            no_split_module_classes = ["LlamaDecoderLayer"]
        elif "mistral" in model_name:
            no_split_module_classes = ["MistralDecoderLayer"]
        else:
            raise NotImplementedError(
                "No distributed inference support for this model."
            )

        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes,
            dtype=model_dtype,
            max_memory=max_memory,
        )

    if 'flan-t5' in model_path.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=model_dtype,
            token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=model_dtype,
            token=hf_token,
        )

    return model

def prepare_model(configs, token, devices):
    # collections = {}
    # for key in configs:
    #     print(key)
    #     if isinstance(configs[key], str) and (configs[key] in configs.keys()):
    #         collections[key] = collections[configs[key]]
    #     elif isinstance(configs[key], dict):
    model_path = configs['path']
    if devices is not None and not isinstance(devices, list):
        devices = [devices]
    
    if devices is None or len(devices) == 1:
        device_map = torch.device(devices[0]) if devices is not None else torch.device("cpu")
        if 'Mistral-7B-Instruct-v0.2' in model_path.lower():
            # Manual setting of sliding window
            # See https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/56
            model_config = AutoConfig.from_pretrained(model_path)
            model_config.update({'sliding_window': 4096})
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map, token=token, config=model_config).eval()
        elif 'flan-t5' in model_path.lower():
            # Not supported with AutoModelForCausalLM
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map, token=token).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map, token=token).eval()
    else:
        model = _load_distributed_hf(configs, devices, hf_token=token).eval()
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(configs['path'], padding_side='left', use_fast=use_fast_tokenizer, token=token)
#         tokenizer.pad_token = tokenizer.eos_token # TODO: check if this is correct?
    for p in model.parameters():
        p.requires_grad_(False)
    collections = {
        'name' : configs['name'],
        'model': model,
        'tokenizer': tokenizer,
    }
    #     else:
    #         raise ValueError
    return EasyDict(collections)

def prepare_data(configs, token, collections):
    
    if configs['name'] == 'toxic-chat':
        if configs['path'] is not None:
            dataset = load_dataset(configs['path'], "toxicchat0124")
        else:
            dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", token=token)

        datas_tt = EasyDict({})
        for split_name in ['train', 'test']: # train, test
            print(split_name)
            datas = dataset[split_name]

            # labels
            y_toxicity = torch.tensor([d['toxicity'] for d in datas]).bool()
            y_jailbreaking = torch.tensor([d['jailbreaking'] for d in datas]).bool()
            assert y_jailbreaking.logical_and(y_toxicity.logical_not()).sum() == 0
            
            print(f"Total number:\t\t{len(datas)}")
            print(f"Toxic number:\t\t{y_toxicity.sum().item()}")
            print(f"jailbreaking number:\t{y_jailbreaking.sum().item()}")
            
            content = [d['user_input'] for d in datas]

            logits_path = os.path.join(configs['logits_dir'], collections.name, f"toxic-chat/results_first_logits_{split_name}.pts")
            if not os.path.exists(logits_path):
                prepare_logits(configs, dataset, collections)
            results_tt = torch.load(logits_path)
            
            datas_tt[split_name] = EasyDict({
                'content': content,
                'toxicity_label': y_toxicity,
                'jailbreaking_label': y_jailbreaking,
                'logits': results_tt,
            })
            print([key for key in datas_tt])
        return datas_tt

    elif configs['name'] == 'lmsys-chat-1m':
        if configs['path'] is not None:
            dataset = load_dataset(configs['path'])
        else:
            dataset = load_dataset("lmsys/lmsys-chat-1m", token=token)

        logits_path = os.path.join(configs['logits_dir'], collections.name, "lmsys-chat-1m/results_first_logits_0to20000.pts")
        if not os.path.exists(logits_path):
            prepare_logits(configs, dataset, collections)
        results_tt = torch.load(logits_path)
        results_tt = results_tt.squeeze()

        select = configs['select']
        dataset_sub = dataset['train'][:select]
        train_num = int(0.5 * select)
        train_ids = np.concatenate([np.ones(train_num), np.zeros(select - train_num)])
        np.random.shuffle(train_ids)
        test_ids = np.logical_not(train_ids)

        # selected = np.array(dataset_sub['turn']) == 1
        selected_ids = np.array(dataset_sub['turn']) > 0
        train_ids = np.logical_and(train_ids, selected_ids)
        test_ids = np.logical_and(test_ids, selected_ids)
        split_ids = {
            'train': train_ids,
            'test': test_ids,
        }
        label_names = [k for k in dataset_sub['openai_moderation'][0][0]['categories']]
        print('\n'.join([f"{i}: {name}" for i, name in enumerate(label_names)]))
        full_scores = np.array([[x[0]['category_scores'][k] for k in label_names] for x in dataset_sub['openai_moderation']])
        full_labels = np.array([[x[0]['categories'][k] for k in label_names] for x in dataset_sub['openai_moderation']]) # logical_or of all labels
        # full_labels = full_scores >= 0.1

        datas_tt = EasyDict({})
        for split_name in ['train', 'test']: # train, test
            print(split_name)

            # labels
            y_full = torch.from_numpy(full_labels[split_ids[split_name]])
            y_score = torch.from_numpy(full_scores[split_ids[split_name]])
            y_toxicity = y_full.sum(1) > 0
        #     y_jailbreaking = torch.tensor([d['jailbreaking'] for d in datas]).bool()
        #     assert y_jailbreaking.logical_and(y_toxicity.logical_not()).sum() == 0
            
            print(f"Total number:\t\t{len(y_toxicity)}")
            print(f"Toxic number:\t\t{y_toxicity.sum().item()}")
        #     print(f"jailbreaking number:\t{y_jailbreaking.sum().item()}")
            
            content = np.array([x[0]['content'] for x in dataset_sub['conversation']])[split_ids[split_name]].tolist()
            
            datas_tt[split_name] = EasyDict({
                'content': content,
                'toxicity_label': y_toxicity,
                'full_label': y_full,
                'full_score': y_score,
                'sentence_ids': train_ids if split_name == 'train' else test_ids,
        #         'jailbreaking_label': y_jailbreaking,
                'logits': results_tt[:select][torch.from_numpy(split_ids[split_name])],
            })
        return datas_tt

    elif configs['name'] == 'lmsys-chat-1m-handlabeled-split':
        with open(configs['path']) as f:
            dataset = json.load(f)
        datas_tt = EasyDict({})
        for split_name in ['train', 'test']:
            print(split_name)

            y_toxicity = torch.tensor([t[2] for t in dataset[split_name]]).bool()
            
            print(f"Total number:\t\t{len(dataset[split_name])}")
            print(f"Toxic number:\t\t{y_toxicity.sum().item()}")
            
            content = [t[1] for t in dataset[split_name]]

            logits_path = os.path.join(configs['logits_dir'], collections.name, f"lmsys-chat-1m-handlabeled-split/results_first_logits_{split_name}.pts")
            if not os.path.exists(logits_path):
                prepare_logits(configs, dataset, collections)
            results_tt = torch.load(logits_path)
            
            datas_tt[split_name] = EasyDict({
                'content': content,
                'toxicity_label': y_toxicity,
                'logits': results_tt,
            })
            print([key for key in datas_tt])
        return datas_tt

    else:
        raise NotImplementedError
    
    
def computeMetrics(scores, FPR, ths=(1e-3, 1e-4, 1e-5)):
    scores1, scores0 = scores

    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    scores = np.concatenate((scores0, scores1))
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    fpr, tpr, thr = roc_curve(labels, scores)
    TPR = np.interp(FPR, fpr, tpr)

    TPRs_lowFPR_interp = np.interp(ths, fpr, tpr)
    TPRs_lowFPR = [tpr[(fpr <= th).nonzero()[0][-1]] for th in ths]
    TPRs_lowFPR = np.array(TPRs_lowFPR)

    # FPR @TPR95
    FPRs = np.interp([.80, .85, .90, .95], tpr, fpr)

    # AUROC
    AUROC = roc_auc(labels, scores)

    # Optimal Accuracy
    # AccList = [accuracy_score(scores > t, labels) for t in thr]
    AccList = 1 - np.logical_xor(scores[:, np.newaxis] > thr[np.newaxis, :], labels[:, np.newaxis]).sum(0) / len(scores)
    Acc_opt = np.max(AccList)
    ind = np.argmax(AccList)
    thr_opt = thr[ind]

    metrics = {
        'AUROC': AUROC,
        'acc_list': AccList,
        'acc_opt': Acc_opt,
        'thr_opt': thr_opt,
        'fpr': fpr,
        'tpr': tpr,
        'thr': thr,
        'FPRs_4': FPRs,
        'TPRs_lowFPR_interp': TPRs_lowFPR_interp,
        'TPRs_lowFPR': TPRs_lowFPR,
        'FPR': FPR,
        'TPR': TPR,
    }
    return metrics


def prepare_logits(configs, dataset, collections):
    if configs['name'] in ['toxic-chat', 'lmsys-chat-1m-handlabeled-split']:
        for split_name in ['train', 'test']: # train, test
            print(split_name)
            datas = dataset[split_name]
            results = []
            for d in tqdm(datas):
                if configs['name'] == 'toxic-chat':
                    content = d['user_input']
                elif configs['name'] == 'lmsys-chat-1m-handlabeled-split':
                    content = d[1]

                model, tokenizer = collections.model, collections.tokenizer
                conv = get_conversation_template(collections.name)

                conv.append_message(conv.roles[0], content)
                conv.append_message(conv.roles[1], None)
                x_init = conv.get_prompt()
                x = x_init + ' '

                tokenized = tokenizer(x, truncation=True, max_length=1024) if collections.name == 'gpt2' else tokenizer(x)
                input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(0).to(model.device)
                # input_ids_m = input_ids
                # logits = model(input_ids=input_ids_m).logits

                if 'flan-t5' in collections.name:
                    generation = model.generate(input_ids=input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                    logits = generation.scores[0]
                else:
                    logits = model(input_ids=input_ids).logits[0, -1]
                results.append(logits.detach().cpu())

            results = torch.stack(results).squeeze()
            
            logits_path = os.path.join(configs['logits_dir'], collections.name, configs['name'], f"results_first_logits_{split_name}.pts")
            if not os.path.exists(os.path.dirname(logits_path)):
                os.makedirs(os.path.dirname(logits_path))
            torch.save(results, logits_path)


    elif configs['name'] == 'lmsys-chat-1m':
        results = []
        for i, d in tqdm(zip(range(20000), dataset['train'])):
            content = d['conversation'][0]['content']
            assert d['conversation'][0]['role'] == 'user'

            model, tokenizer = collections.model, collections.tokenizer
            conv = get_conversation_template(collections.name)
            conv.append_message(conv.roles[0], content)
            conv.append_message(conv.roles[1], None)
            x_init = conv.get_prompt()
            x = x_init + ' '

            # TODO: use batch
            tokenized = tokenizer(x, truncation=True, max_length=1024) if collections.name == 'gpt2' else tokenizer(x)
            input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(0).to(model.device)
            model_inputs = model.prepare_inputs_for_generation(input_ids)
            if 'flan-t5' in collections.name:
                generation = model.generate(input_ids=input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                logits = generation.scores[0]
            else:
                logits = model(input_ids=input_ids).logits[0, -1]

            results.append(logits.detach().cpu())
            
        results = torch.stack(results).squeeze()
        logits_path = os.path.join(configs['logits_dir'], collections.name, "lmsys-chat-1m/results_first_logits_0to20000.pts")
        if not os.path.exists(os.path.dirname(logits_path)):
            os.makedirs(os.path.dirname(logits_path))
        torch.save(results, logits_path)

    else:
        raise NotImplementedError