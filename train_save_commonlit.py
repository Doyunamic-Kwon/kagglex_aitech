# -*- coding: utf-8 -*-
# CommonLit DistilBERT 회귀 모델을 로컬에서 학습 -> Kaggle 데이터셋으로 올릴 폴더에 저장
import os, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available(): torch.mps.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

MODEL_NAME, MAX_LEN, BATCH, EPOCHS, LR = "distilbert-base-uncased", 256, 16, 4, 2e-5
OUT = "kaggle_ds_model/model"
os.makedirs(OUT, exist_ok=True)

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

train = pd.read_csv("data_commonlit/train.csv")
tr_idx, va_idx = train_test_split(train.index, test_size=0.15, random_state=SEED)
y_tr = train.loc[tr_idx, "target"].values; y_va = train.loc[va_idx, "target"].values

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def enc(texts): return tok(list(texts), truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
e_tr, e_va = enc(train.loc[tr_idx,"excerpt"]), enc(train.loc[va_idx,"excerpt"])
tl = DataLoader(TensorDataset(e_tr["input_ids"], e_tr["attention_mask"], torch.tensor(y_tr, dtype=torch.float)), batch_size=BATCH, shuffle=True)
vl = DataLoader(TensorDataset(e_va["input_ids"], e_va["attention_mask"], torch.tensor(y_va, dtype=torch.float)), batch_size=32)

@torch.no_grad()
def pred(m, loader):
    m.eval(); out=[]
    for b in loader:
        out.append(m(input_ids=b[0].to(device), attention_mask=b[1].to(device)).logits.squeeze(-1).cpu())
    return torch.cat(out).numpy()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, int(0.1*len(tl)*EPOCHS), len(tl)*EPOCHS)

best, best_state = 1e9, None
for ep in range(1, EPOCHS+1):
    model.train(); t0=time.time()
    for ids, attn, lab in tl:
        ids, attn, lab = ids.to(device), attn.to(device), lab.to(device)
        loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
    r = rmse(y_va, pred(model, vl))
    print(f"[epoch {ep}] val_RMSE={r:.4f} ({time.time()-t0:.0f}s)")
    if r < best: best, best_state = r, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)
print("BEST val RMSE:", round(best,4))
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print("saved model to", OUT)
print("files:", os.listdir(OUT))
