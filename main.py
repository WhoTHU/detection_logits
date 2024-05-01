import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from tqdm.auto import tqdm

from utils import *
import configs
device = torch.device(configs.device)
########## Load model
collections = prepare_model(configs.model_configs, configs.hf_token, device)
########## Load and prepare dataset
datas_tt = prepare_data(configs.dataset_configs, configs.hf_token, collections)
########## Regression
split_name = 'train' # train, test
content = datas_tt[split_name]['content']
y_toxicity = datas_tt[split_name]['toxicity_label']
results_tt = datas_tt[split_name]['logits']

def get_feature(results_tt):
    results_tt = results_tt.softmax(1)
#     results_tt = results_tt.log()
    results_tt = results_tt.log() - (1 - results_tt).log()
#     x = torch.zeros_like(results_tt)
#     idx = results_tt.topk(1000, 1)[1]
#     t = results_tt.gather(1, idx)
#     x.scatter_(1, idx, t)
#     results_tt = x
    return results_tt
results_tt = get_feature(results_tt)

from torch.utils.data import TensorDataset, DataLoader

class LogisticRegression(torch.nn.Module):    
    def __init__(self, n_inputs, n_outputs, normalization=[None, None]):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.register_buffer('mean', normalization[0])
        self.register_buffer('std', normalization[1] + 1e-10)
    def forward(self, x):
        if self.mean is not None:
            x = (x - self.mean) / self.std
        y_pred = self.linear(x)
        return y_pred

train_data = TensorDataset(results_tt, y_toxicity.float())
train_loader = DataLoader(train_data, batch_size=128)
mean_fixed = results_tt.mean(0)
std_fixed = results_tt.std(0)

n_inputs = 32000
n_outputs = 1
log_regr = LogisticRegression(n_inputs, n_outputs, normalization=(mean_fixed, std_fixed)).to(device)
### unbalanced sample weights
pos_weight = y_toxicity.logical_not().sum() / y_toxicity.sum()

optimizer = torch.optim.SGD(log_regr.parameters(), lr=5e-4) #4e-3 for logits, 5e-4 for logp/1-p
# optimizer = torch.optim.Adam(log_regr.parameters(), lr=0.001)

epochs = 200
losses = []
for epoch in range(epochs):
    for i, (x, y) in tqdm(enumerate(train_loader)):
        x, y  = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = log_regr(x)[..., 0]
        loss = F.binary_cross_entropy_with_logits(outputs, y, pos_weight=pos_weight)
        paras = torch.cat([p.view(-1) for p in log_regr.linear.parameters()])
        loss = loss + 1e-3 * torch.norm(paras, 1) # L1 loss
#         loss = loss + 1e-1 * torch.norm(paras, 2)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
losses = np.array(losses)
np.save('./cache/losses.npy', losses)
torch.save(log_regr.state_dict(), './cache/regression.pt')

# ########## Plot train
# print(f"Epoch: {epoch}")
# plt.figure()
# plt.plot(losses)
# plt.show()

# plt.figure()
# density = True
# w = log_regr.linear.weight.detach()[0].cpu()
# f = plt.hist(w.numpy(), 100, density=True);
# plt.hist([w[306].numpy()], bins=f[1], weights=[np.max(f[0])], alpha=0.5);
# plt.hist([w[8221].numpy()], bins=f[1], weights=[np.max(f[0])], alpha=0.5);
# plt.legend(['All', 'I', 'Sorry'])
# plt.show()

# ntopk = 20
# wtopkids = w.topk(ntopk)[1]
# for i in wtopkids:
#     print(f"Id is {i}: {tokenizer.decode(i)}")

# plt.hist(results_tt.flatten().numpy(), 100, density=density, alpha=1/3);
# plt.hist(results_tt[y_toxicity.logical_not()][:, wtopkids].flatten().numpy(), 100, density=density, alpha=1/3);
# plt.hist(results_tt[y_toxicity][:, wtopkids].flatten().numpy(), 100, density=density, alpha=1/3);
# plt.legend(['ALL', 'non-toxic', 'toxic'])
# plt.show()

########## Test
split_name = 'test' # train, test
content = datas_tt[split_name]['content']
y_toxicity = datas_tt[split_name]['toxicity_label']
results_tt = datas_tt[split_name]['logits']

results_tt = get_feature(results_tt)
results = log_regr(results_tt.to(device))[..., 0].detach().cpu()
# results = results_tt[:, 306] # logits
# results = results_tt[:, 8221] # logits

r1 = results[y_toxicity.logical_not()].numpy()
r2 = results[y_toxicity].numpy()

scores_split = [r2, r1]
FPR = np.linspace(0, 1, 1001)
ths = (1e-1, 1e-2, 1e-3, 1e-4)
metrics = computeMetrics(scores_split, FPR, ths)
np.save('./cache/metrics', metrics)
np.savez('./cache/scores_split', r1=r1, r2=r2)

y_score = np.concatenate(scores_split)
y_test = np.concatenate([np.ones_like(scores_split[0]), np.zeros_like(scores_split[1])])
average_precision = average_precision_score(y_test, y_score)
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
auprc = auc(recall, precision)

print(f"AUPRC is {auprc}")

# ########## Plot test
# plt.figure(figsize=(12, 7.5))
# plt.subplot(2,2,1)
# for sc in scores_split:
#     plt.hist(sc, 100, density=True, alpha=0.5);
# #     plt.xlim([-0.02, 1.02])
# #     plt.ylim([0, 100])
# plt.legend(['positives', 'negatives'])
# plt.title('AUROC is %.4f, success rate is %.4f with threshold %.4f' % (metrics['AUROC'], metrics['acc_opt'], metrics['thr_opt'], ))

# plt.subplot(2,2,2)
# plt.plot(metrics['FPR'], metrics['TPR'])
# plt.plot([0, 1], [0, 1], 'k--')

# plt.subplot(2,2,3)
# x = np.log(metrics['FPR'])
# y = np.log(metrics['TPR'])
# plt.plot(x, y)
# mi = min(x[x>-np.Inf].min(), y[y>-np.Inf].min())
# ma = max(y[x>-np.Inf].max(), y[y>-np.Inf].max())
# plt.plot([mi, ma], [mi, ma], 'k--')
# plt.title(f"TPRs at low FPR are " + ', '.join([f"{x:.2f}" for x in metrics['TPRs_lowFPR'] * 100]))
# plt.xlabel('FPR')
# plt.ylabel('TPR')

# plt.subplot(2,2,4)
# y_score = np.concatenate(scores_split)
# y_test = np.concatenate([np.ones_like(scores_split[0]), np.zeros_like(scores_split[1])])
# average_precision = average_precision_score(y_test, y_score)
# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# auprc = auc(recall, precision)
# plt.plot(recall, precision)
# plt.title(f"AP is {average_precision:.2%}, AUPRC is {auprc:.2%}")
# plt.xlabel('Recall')
# plt.ylabel('Precision')

# plt.show()
# {k: metrics[k] if not k in ['acc_list', 'thr', 'tpr', 'fpr', 'FPR', 'TPR'] else 'collapsed' for k in metrics.keys()}



