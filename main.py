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
model_name = configs.model_configs['target']['name'] # 'llama-3', 'mistral-7b', etc.
########## Load and prepare dataset
data_dir = os.path.join(configs.dataset_configs['logits_dir'], model_name) # './cache' - TODO: maybe rename this?
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
datas_tt = prepare_data(configs.dataset_configs, configs.hf_token, collections)
########## Regression
split_name = 'train' # train, test
content = datas_tt[split_name]['content']
y_toxicity = datas_tt[split_name]['toxicity_label']
results_tt = datas_tt[split_name]['logits']

EPOCHS = 384
BATCH_SIZE = 32
LEARNING_RATE = 4e-4 #4e-3 for logits, 5e-4 for logp/1-p
L1_REG = 4e-4
TRAIN_NUM = datas_tt.train.toxicity_label.shape[0]
TEST_NUM = datas_tt.test.toxicity_label.shape[0]
params_repr = f"E{EPOCHS}_B{BATCH_SIZE}_LR{'{:.1e}'.format(LEARNING_RATE)}_L1R{'{:.1e}'.format(L1_REG)}_TRN{TRAIN_NUM}_TEST{TEST_NUM}"
if not os.path.exists(os.path.join(data_dir, params_repr)):
    os.makedirs(os.path.join(data_dir, params_repr))

def get_feature(results_tt):
    if len(results_tt.shape) == 1:
        results_tt = results_tt.unsqueeze(1)
    results_tt = results_tt.to(torch.float32).softmax(1)
#     results_tt = results_tt.log()
    # Bound away from 0 and 1 to avoid numerical instability
    results_tt = torch.clamp(results_tt, min=1e-16).log() - torch.clamp(1 - results_tt, min=1e-16).log()
#     x = torch.zeros_like(results_tt)
#     idx = results_tt.topk(1000, 1)[1]
#     t = results_tt.gather(1, idx)
#     x.scatter_(1, idx, t)
#     results_tt = x
    return results_tt
results_tt = get_feature(results_tt)

from torch.utils.data import TensorDataset, DataLoader

class LogisticRegression(torch.nn.Module):    
    def __init__(self, n_inputs, n_outputs, normalization=[None, None], min_std=1e-10):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.min_std = min_std
        self.register_buffer('mean', normalization[0])
        self.register_buffer('std', normalization[1] + min_std)
    def forward(self, x):
        if self.mean is not None:
            x = (x - self.mean) / self.std
        # Ignore deterministic tokens
        x = x * (self.std > self.min_std)
        y_pred = self.linear(x)
        return y_pred

train_data = TensorDataset(results_tt, y_toxicity.float())
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
mean_fixed = results_tt.mean(0)
std_fixed = results_tt.std(0)

n_inputs = collections.target.model.lm_head.out_features
n_outputs = 1
log_regr = LogisticRegression(n_inputs, n_outputs, normalization=(mean_fixed, std_fixed)).to(device)
### unbalanced sample weights
pos_weight = y_toxicity.logical_not().sum() / y_toxicity.sum()

optimizer = torch.optim.SGD(log_regr.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.Adam(log_regr.parameters(), lr=0.001)

losses = []
for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        x, y  = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = log_regr(x)[..., 0]
        loss = F.binary_cross_entropy_with_logits(outputs, y, pos_weight=pos_weight)
        paras = torch.cat([p.view(-1) for p in log_regr.linear.parameters()])
        loss = loss + L1_REG * torch.norm(paras, 1) # L1 loss
#         loss = loss + 1e-1 * torch.norm(paras, 2)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
losses = np.array(losses)
np.save(os.path.join(data_dir, params_repr, 'losses.npy'), losses)
torch.save(log_regr.state_dict(), os.path.join(data_dir, params_repr, 'regression.pt'))

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

# Split the results into batches
num_batches = (results_tt.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE

results = []
for i in range(num_batches):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, results_tt.shape[0])
    batch_results = results_tt[start_idx:end_idx].to(device)
    batch_outputs = log_regr(batch_results)[..., 0].detach().cpu()
    results.append(batch_outputs)

results = torch.cat(results)

r1 = results[y_toxicity.logical_not()].numpy()
r2 = results[y_toxicity].numpy()

scores_split = [r2, r1]
FPR = np.linspace(0, 1, 1001)
ths = (1e-1, 1e-2, 1e-3, 1e-4)
metrics = computeMetrics(scores_split, FPR, ths)
np.save(os.path.join(data_dir, params_repr, 'metrics'), metrics)
np.savez(os.path.join(data_dir, params_repr, 'scores_split'), r1=r1, r2=r2)

y_score = np.concatenate(scores_split)
y_test = np.concatenate([np.ones_like(scores_split[0]), np.zeros_like(scores_split[1])])
average_precision = average_precision_score(y_test, y_score)
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
auprc = auc(recall, precision)

print(f"AUPRC is {auprc}")

# Print parameters
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, L1 regularization: {L1_REG}")
print(f"Train size: {TRAIN_NUM}, Test size: {TEST_NUM}")

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



