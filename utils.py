import os
import torch
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as roc_auc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from easydict import EasyDict
from tqdm.auto import tqdm
from fastchat.model import get_conversation_template
import json


def prepare_model(configs, token, device):
    collections = {}
    for key in configs:
        print(key)
        if isinstance(configs[key], str) and (configs[key] in configs.keys()):
            collections[key] = collections[configs[key]]
        elif isinstance(configs[key], dict):
            model = AutoModelForCausalLM.from_pretrained(configs[key]['path'], torch_dtype=torch.float16, device_map=device, token=token).eval()
            use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
            tokenizer = AutoTokenizer.from_pretrained(configs[key]['path'], padding_side='left', use_fast=use_fast_tokenizer, token=token)
    #         tokenizer.pad_token = tokenizer.eos_token
            for p in model.parameters():
                p.requires_grad_(False)
            collections[key] = {
                'name' : configs[key]['name'],
                'model': model,
                'tokenizer': tokenizer,
            }
        else:
            raise ValueError
    return EasyDict(collections)

def prepare_data(configs, token, collections):
    
    if configs['name'] == 'toxic-chat':
        if configs['path'] is not None:
            dataset = load_dataset(configs['path'])
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

            logits_path = os.path.join(configs['logits_dir'], f"toxic-chat/results_first_logits_{split_name}.pts")
            if not os.path.exists(logits_path):
                prepare_logits(configs, dataset, collections)
            results_tt = torch.load(logits_path)
            
            datas_tt[split_name] = EasyDict({
                'content': content,
                'toxicity_label': y_toxicity,
                'jailbreaking_label': y_jailbreaking,
                'logits': results_tt,
            })
            print(f"Total number:\t\t{len(dataset[split_name])}")
            print(f"Toxic number:\t\t{datas_tt[split_name]['toxicity_label'].sum().item()}")
        return datas_tt

    elif configs['name'] == 'lmsys-chat-1m':
        if configs['path'] is not None:
            dataset = json.load(open(configs['path'], 'r'))
        else:
            print('Please download manually labeled data')
            raise ValueError
        
        datas_tt = EasyDict({})
        for split_name in ['train', 'test']: # train, test
            print(split_name)
            logits_path = os.path.join(configs['logits_dir'], f"lmsys-chat-1m/results_first_logits_{split_name}.pts")
            if not os.path.exists(logits_path):
                prepare_logits(configs, dataset, collections)
            results_tt = torch.load(logits_path)

            datas_tt[split_name] = EasyDict({
                'content': [d[1] for d in dataset[split_name]],
                'toxicity_label': torch.tensor([d[2] for d in dataset[split_name]]).bool(),
                'jailbreaking_label': torch.tensor([d[4] for d in dataset[split_name]]).bool(),
                'logits': results_tt,
            })
            print(f"Total number:\t\t{len(dataset[split_name])}")
            print(f"Toxic number:\t\t{datas_tt[split_name]['toxicity_label'].sum().item()}")
        return datas_tt

    else:
        raise NotImplementederror
    
    
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
    if configs['name'] in ['toxic-chat', 'lmsys-chat-1m']:
        for split_name in ['train', 'test']: # train, test
            print(split_name)
            datas = dataset[split_name]
            results = []
            for d in tqdm(datas):
                content = d['user_input'] if configs['name'] == 'toxic-chat' else d[1]

                target_collection = collections.target
                model, tokenizer = target_collection.model, target_collection.tokenizer
                conv = get_conversation_template(target_collection.name)

                conv.append_message(conv.roles[0], content)
                conv.append_message(conv.roles[1], None)
                x_init = conv.get_prompt()
                x = x_init + ' '

                input_ids = torch.tensor(tokenizer(x)['input_ids']).unsqueeze(0).to(model.device)
                input_ids_m = input_ids
                logits = model(input_ids=input_ids_m).logits

                l = logits[0, -1]
                results.append(l.detach().cpu())

            results = torch.stack(results)
            
            logits_path = os.path.join(configs['logits_dir'], f"{configs['name']}/results_first_logits_{split_name}.pts")
            if not os.path.exists(os.path.dirname(logits_path)):
                os.makedirs(os.path.dirname(logits_path))
            torch.save(results, logits_path)
    else:
        raise NotImplementederror