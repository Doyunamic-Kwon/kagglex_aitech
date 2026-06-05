#!/usr/bin/env python
# coding: utf-8

# # 📚 CommonLit Readability Prize — 글의 '읽기 난이도' 예측 (회귀)
# 
# > 목표: 짧은 영어 지문(excerpt)이 **얼마나 읽기 쉬운지**를 나타내는 `target` 점수를 예측.
# > target이 **낮을수록 어려운 글**, 높을수록 쉬운 글. 평가지표는 대회 기준 **RMSE**.
# 
# 이건 재난 트윗 과제(이진 분류)와 달리 **연속값을 맞히는 회귀** 문제라, 같은 3단계 틀을 회귀에 맞게 변형:
# 1. **EDA** — target 분포, 지문 길이, standard_error, 결측 확인
# 2. **Baseline** — 전처리 → **TF-IDF → Ridge 회귀** (분류의 LogReg 자리에 회귀모델)
# 3. **(추가 실험)** 고전적 **가독성 피처(Flesch 등)** 가 도움이 되는지 ablation
# 4. **개선** — **DistilBERT 회귀 헤드** 파인튜닝 (MPS 가속)
# 5. **분석** — RMSE, **예측 vs 실제 산점도**, **잔차 분포**, **과대/과소예측 최악 사례** 분석
#    (분류의 Confusion Matrix·FP/FN 에 대응되는 회귀용 진단)
# 6. **제출** — test 예측 → submission_commonlit.csv
# 
# > 메모: test가 7행밖에 없다(코드대회라 실제 채점셋은 숨겨져 있음). 그래서 제출파일은 7개 예측만 나옴.
# > 모델 성능 판단은 내부 validation RMSE로 한다.

# In[1]:


import os, re, random, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 140)

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    except Exception: pass
seed_everything()

DATA_DIR = "data_commonlit"
FIG_DIR  = "figures_commonlit"
os.makedirs(FIG_DIR, exist_ok=True)

def rmse(y_true, y_pred):   # 평가지표, 함수로
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
print("setup done.")


# ## 0. 데이터 로드 & EDA

# In[2]:


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print("train:", train.shape, "| test:", test.shape)
print("columns:", list(train.columns))
print("\n결측치:\n", train.isna().sum())
# url/license는 70% 결측, 출처 메타라 안 씀. 쓸 건 excerpt랑 target
train.head(2)


# In[3]:


# target 분포. 회귀라 모양이 중요(치우침/꼬리)
print(train["target"].describe().round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
sns.histplot(train["target"], bins=40, kde=True, ax=axes[0], color="#4C78A8")
axes[0].set_title("target distribution (lower = harder to read)")
sns.histplot(train["standard_error"], bins=40, ax=axes[1], color="#E45756")
axes[1].set_title("standard_error of target")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/target_dist.png", dpi=120); plt.show()
# target 종모양(평균 -0.96). 치우침 없어 변환 안 함. standard_error는 라벨 불확실성


# In[4]:


# 길이랑 난이도 상관 있나
train["n_chars"] = train["excerpt"].str.len()
train["n_words"] = train["excerpt"].str.split().apply(len)
print("길이 통계:\n", train[["n_chars","n_words"]].describe().loc[["min","mean","max"]].round(0))
print("\ntarget과 상관계수:")
print(train[["n_chars","n_words","target"]].corr()["target"].round(3))

fig, ax = plt.subplots(figsize=(5,3.4))
ax.scatter(train["n_words"], train["target"], s=8, alpha=0.3)
ax.set_xlabel("word count"); ax.set_ylabel("target"); ax.set_title("length vs readability")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/len_vs_target.png", dpi=120); plt.show()
# 길이 거의 일정(일부러 자른 듯). 난이도랑 상관 거의 없음 -> 어휘/구조를 봐야 함


# ### EDA 메모
# - 예측에 쓸 핵심 컬럼은 사실상 **`excerpt`(지문)** 하나. (url/license는 메타라 제외)
# - target은 치우침 없는 종모양 → 변환 없이 회귀 진행.
# - 지문 길이는 일부러 비슷하게 맞춰져 있어 **길이로는 난이도를 못 가른다** → 어휘/문장 복잡도가 핵심.

# ## 1. Baseline: 전처리 → TF-IDF → Ridge 회귀
# 
# > 분류의 LogReg 자리에, 회귀에선 **Ridge**(L2 정규화 선형회귀)가 텍스트 회귀 baseline으로 무난.
# > 같은 검증셋에서 비교하려고 train/val 분할을 먼저 고정.

# In[5]:


# 줄바꿈/특수문자만 정리. 단어 정보는 최대한 보존
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)          # 공백 정리
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\;\:']", " ", s)  # 문장부호 일부 남김(난이도 신호)
    s = re.sub(r"\s+", " ", s).strip()
    return s

train["clean"] = train["excerpt"].map(clean_text)
test["clean"]  = test["excerpt"].map(clean_text)

from sklearn.model_selection import train_test_split
tr_idx, va_idx = train_test_split(train.index, test_size=0.15, random_state=SEED)
y_tr = train.loc[tr_idx, "target"].values
y_va = train.loc[va_idx, "target"].values
print(f"train: {len(tr_idx)} / val: {len(va_idx)}")


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

def make_tfidf():
    # min_df=3, 희귀 토큰은 과적합
    return TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)

X_text = train["clean"]
kf = KFold(5, shuffle=True, random_state=SEED)

# alpha 훑기. 텍스트 회귀는 약한 정규화가 보통 나음
print("=== Ridge alpha 튜닝 (5-fold CV RMSE) ===")
best_alpha, best_cv = None, 1e9
for alpha in [0.3, 1.0, 2.0, 5.0]:
    pipe = Pipeline([("tfidf", make_tfidf()), ("ridge", Ridge(alpha=alpha))])
    neg = cross_val_score(pipe, X_text, train["target"], cv=kf, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-neg).mean()
    print(f"  alpha={alpha:4.1f} -> CV RMSE = {cv_rmse:.4f}")
    if cv_rmse < best_cv: best_cv, best_alpha = cv_rmse, alpha
print("-> best alpha:", best_alpha)

# 고정 val 점수(비교용)
base_pipe = Pipeline([("tfidf", make_tfidf()), ("ridge", Ridge(alpha=best_alpha))])
base_pipe.fit(X_text.loc[tr_idx], y_tr)
pred_base_val = base_pipe.predict(X_text.loc[va_idx])
print(f"\nbaseline val RMSE = {rmse(y_va, pred_base_val):.4f}")
print(f"(참고) 평균만 찍는 naive RMSE = {rmse(y_va, np.full_like(y_va, y_tr.mean())):.4f}")


# ### Baseline 메모
# - TF-IDF + Ridge로 CV RMSE ~0.72 수준. 평균만 찍는 naive(~1.03)보다 확실히 좋음 → 단어 빈도만으로도 난이도 신호가 꽤 있음.
# - 그래도 0.72는 그리 좋은 점수는 아님(상위권은 0.45 근처). 선형모델은 **어순/문맥**을 못 봐서 한계가 분명.
# - 다음: "사람이 만든 가독성 지표"를 넣으면 도움이 될지 → 그 다음 DistilBERT.

# ## 1.5 (추가 실험) 고전적 가독성 피처가 도움이 될까?
# 
# > 가독성에는 이미 수십 년 된 공식들이 있다 (Flesch Reading Ease, 평균 문장 길이, 음절 수 등).
# > 가설: *이런 구조적 지표를 TF-IDF에 더하면 선형모델이 못 보던 '문장 복잡도'를 보강해줄 것이다.*
# > (재난 과제의 '엔티티 피처 실험'과 같은 취지 — baseline 위에 얹어 효과를 정직하게 검증.)

# In[7]:


# textstat 없이 직접. 가독성 지표 손으로 뽑음
def count_syllables(word):
    # 음절 근사: 모음 그룹 수 + 끝 묵음 e 보정
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1: n -= 1
    return max(n, 1) if re.search(r"[a-z]", word) else 0

def readability_features(text):
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    n_sent = max(len(sents), 1); n_word = max(len(words), 1)
    syl = sum(count_syllables(w) for w in words)
    long_words = sum(1 for w in words if len(w) > 6)
    uniq = len(set(w.lower() for w in words))
    flesch = 206.835 - 1.015 * (n_word / n_sent) - 84.6 * (syl / n_word)  # 높을수록 쉬움
    return {
        "r_avg_sent_len": n_word / n_sent,         # 문장당 단어수
        "r_avg_word_len": np.mean([len(w) for w in words]),
        "r_avg_syllable": syl / n_word,            # 단어당 음절
        "r_pct_long":     long_words / n_word,     # 긴 단어 비율
        "r_ttr":          uniq / n_word,           # type-token ratio(어휘 다양성)
        "r_flesch":       flesch,
    }

rf = pd.DataFrame([readability_features(t) for t in train["excerpt"]], index=train.index)
print(rf.describe().round(2).loc[["mean","min","max"]])
print("\n각 가독성 피처와 target 상관:")
print(rf.join(train["target"]).corr()["target"].round(3).sort_values())
# Flesch는 +상관, 문장길이/음절은 -상관. 방향 다 상식대로


# In[8]:


# TF-IDF만 vs +가독성, 같은 val에서 비교
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

tfidf_fitted = make_tfidf().fit(train.loc[tr_idx, "clean"])
scaler = StandardScaler().fit(rf.loc[tr_idx])

def build_X(idx):
    Xt = tfidf_fitted.transform(train.loc[idx, "clean"])
    Xr = csr_matrix(scaler.transform(rf.loc[idx]))
    return hstack([Xt, Xr]).tocsr()

ridge_a = Ridge(alpha=1.0).fit(tfidf_fitted.transform(train.loc[tr_idx, "clean"]), y_tr)
rmse_tfidf = rmse(y_va, ridge_a.predict(tfidf_fitted.transform(train.loc[va_idx, "clean"])))

ridge_b = Ridge(alpha=1.0).fit(build_X(tr_idx), y_tr)
pred_read_val = ridge_b.predict(build_X(va_idx))
rmse_read = rmse(y_va, pred_read_val)

print(f"TF-IDF only          : val RMSE = {rmse_tfidf:.4f}")
print(f"TF-IDF + 가독성 피처 : val RMSE = {rmse_read:.4f}")
print(f"개선폭(작을수록 좋음): {rmse_read - rmse_tfidf:+.4f}")


# ### 가독성 실험 해석 (정직하게)
# - 가독성 피처 각각은 target과 **상식적인 방향의 상관**을 보였다(문장 길수록·음절 많을수록 어렵다 → 점수 낮다). 직관은 맞음.
# - 다만 TF-IDF에 얹었을 때 RMSE 개선은 크지 않다. 이유: TF-IDF가 이미 어휘 수준을 어느 정도 반영하고, 가독성 공식은 6개 숫자라 정보량이 제한적.
# - 결론: 해석엔 좋지만 성능의 한계를 깨는 건 결국 **문맥/어순을 보는 모델**. → DistilBERT로.

# ## 2. 개선: DistilBERT 회귀 헤드 파인튜닝
# 
# > 분류 때와 달리 출력 뉴런 1개(`num_labels=1`)로 두고 **MSE loss**로 점수를 직접 회귀.
# > 지문이 길어서(평균 ~200 토큰) `max_length`를 256으로 키움. 데이터가 작아(2.8k) 과적합 조심.

# In[9]:


import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256    # 토큰 p95=259라 256이면 대부분 커버
BATCH   = 16     # 길어서 32->16
EPOCHS  = 4      # 데이터 작아 조금 더, val로 과적합 감시
LR      = 2e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def encode(texts):
    return tokenizer(list(texts), truncation=True, padding="max_length",
                     max_length=MAX_LEN, return_tensors="pt")

enc_tr = encode(train.loc[tr_idx, "excerpt"])   # BERT엔 원문 그대로
enc_va = encode(train.loc[va_idx, "excerpt"])
train_ds = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], torch.tensor(y_tr, dtype=torch.float))
val_ds   = TensorDataset(enc_va["input_ids"], enc_va["attention_mask"], torch.tensor(y_va, dtype=torch.float))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
print("batches/epoch:", len(train_loader))


# In[10]:


@torch.no_grad()
def predict_reg(model, loader):
    model.eval(); outs = []
    for batch in loader:
        ids, attn = batch[0].to(device), batch[1].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits.squeeze(-1)
        outs.append(logits.cpu())
    return torch.cat(outs).numpy()

seed_everything()
# num_labels=1이면 HF가 알아서 회귀(MSE). labels는 float
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

best_rmse, best_state = 1e9, None
for epoch in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); running = 0.0
    for ids, attn, lab in train_loader:
        ids, attn, lab = ids.to(device), attn.to(device), lab.to(device)
        loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        running += loss.item()
    val_rmse = rmse(y_va, predict_reg(model, val_loader))
    print(f"[epoch {epoch}] train_mse={running/len(train_loader):.4f} | val_RMSE={val_rmse:.4f} | {time.time()-t0:.0f}s")
    if val_rmse < best_rmse:   # RMSE 낮은 시점 저장
        best_rmse = val_rmse
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print("\nbest val RMSE:", round(best_rmse, 4))
model.load_state_dict(best_state)
pred_bert_val = predict_reg(model, val_loader)


# ### DistilBERT 메모
# - baseline(0.72)보다 val RMSE가 내려감 → 문맥/어순을 보는 게 가독성 예측에 실제로 도움.
# - 데이터가 2.8k로 작아서 epoch 늘리면 금방 과적합 → val RMSE 최저 시점 가중치를 저장하는 방식으로 방어.
# - (실무 팁 메모: 여기서 더 가려면 5-fold로 여러 모델 앙상블 + target 정규화가 정석이지만, 과제 범위에선 단일 모델로 충분히 개선 입증.)

# ## 3. 결과 분석: RMSE / 예측-실제 산점도 / 잔차 / 과대·과소예측 사례
# 
# > 회귀라 혼동행렬은 안 맞음. 대신 **예측 vs 실제 산점도**(대각선에 붙을수록 good),
# > **잔차 분포**, 그리고 분류의 FP/FN에 대응하는 **가장 크게 틀린(과대/과소예측) 사례**를 본다.

# In[11]:


summary = pd.DataFrame({
    "model": ["TF-IDF + Ridge (baseline)", "TF-IDF + Readability", "DistilBERT (fine-tuned)"],
    "val_RMSE": [rmse(y_va, pred_base_val), rmse(y_va, pred_read_val), rmse(y_va, pred_bert_val)],
}).sort_values("val_RMSE", ascending=False)

fig, ax = plt.subplots(figsize=(6.5, 3))
bars = ax.barh(summary["model"], summary["val_RMSE"], color=["#9AA0A6", "#6C8EBF", "#54A24B"])
ax.set_xlabel("Validation RMSE (lower is better)")
ax.set_title("Model comparison (same validation split)")
for b, v in zip(bars, summary["val_RMSE"]): ax.text(v+0.005, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/model_compare.png", dpi=120); plt.show()
print(summary.round(4).to_string(index=False))


# In[12]:


# 예측vs실제 + 잔차, 나란히
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, pred, name in zip(axes, [pred_base_val, pred_bert_val], ["TF-IDF+Ridge", "DistilBERT"]):
    ax.scatter(y_va, pred, s=12, alpha=0.4)
    lim = [min(y_va.min(), pred.min()), max(y_va.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1)   # 완벽하면 이 선 위에 다 옴
    ax.set_xlabel("true target"); ax.set_ylabel("predicted")
    ax.set_title(f"{name}  (RMSE={rmse(y_va, pred):.3f})")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/pred_vs_true.png", dpi=120); plt.show()


# In[13]:


# 잔차 분포. 0 중심 좁을수록 good, 쏠림 있나 확인
resid_base = y_va - pred_base_val
resid_bert = y_va - pred_bert_val
fig, ax = plt.subplots(figsize=(6,3.4))
sns.histplot(resid_base, bins=40, color="#9AA0A6", alpha=0.5, label="baseline", ax=ax)
sns.histplot(resid_bert, bins=40, color="#54A24B", alpha=0.5, label="DistilBERT", ax=ax)
ax.axvline(0, color="r", ls="--", lw=1); ax.set_xlabel("residual (true - pred)"); ax.legend()
ax.set_title("Residual distribution")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/residuals.png", dpi=120); plt.show()
print("잔차 표준편차  baseline:", round(resid_base.std(),3), "| DistilBERT:", round(resid_bert.std(),3))
print("잔차 평균(편향) baseline:", round(resid_base.mean(),3), "| DistilBERT:", round(resid_bert.mean(),3))


# In[14]:


# FP/FN 대신 가장 크게 틀린 예측 읽어보기
val_df = train.loc[va_idx, ["excerpt", "target"]].copy()
val_df["pred"]  = pred_bert_val
val_df["error"] = val_df["pred"] - val_df["target"]   # +면 과대(쉽다고 봄), -면 과소

over  = val_df.sort_values("error", ascending=False).head(3)   # 과대예측 worst
under = val_df.sort_values("error").head(3)                    # 과소예측 worst

print("### 과대예측 worst (실제보다 '쉽다'고 잘못 봄):\n")
for _, r in over.iterrows():
    print(f"[true={r.target:+.2f} pred={r.pred:+.2f} err={r.error:+.2f}]")
    print("  ", r.excerpt[:200].replace(chr(10)," "), "...\n")
print("### 과소예측 worst (실제보다 '어렵다'고 잘못 봄):\n")
for _, r in under.iterrows():
    print(f"[true={r.target:+.2f} pred={r.pred:+.2f} err={r.error:+.2f}]")
    print("  ", r.excerpt[:200].replace(chr(10)," "), "...\n")


# ### 잔차/오차 분석 & 인사이트
# 
# - **예측-실제 산점도**: DistilBERT 쪽 점들이 baseline보다 대각선(완벽예측선)에 더 밀착 → 개선이 그림으로도 확인됨.
# - **잔차 분포**: 둘 다 0 중심이라 큰 편향(systematic bias)은 없음. DistilBERT가 더 좁음(오차 분산↓).
# - **최악 사례(과대/과소예측)**: 보통
#   - *과대예측*(실제보다 쉽다고 본 경우): 어휘는 평이한데 **추상적 개념/긴 호흡의 논리**라 사람에겐 어려운 글.
#   - *과소예측*(실제보다 어렵다고 본 경우): 고어체·드문 단어가 섞였지만 내용 자체는 단순한 글. 모델이 '낯선 단어=어려움'으로 과민반응.
# - **개선 방향**: target 정규화 후 회귀, 5-fold 앙상블, 더 긴 max_length, RoBERTa/Longformer. 라벨에 standard_error가 큰 샘플은 가중치를 낮추는 것도 방법.

# ## 4. 제출 파일 생성 (submission_commonlit.csv)

# In[15]:


# test 예측(7행뿐). 포맷 맞춰 저장
enc_te = encode(test["excerpt"])
test_ds = TensorDataset(enc_te["input_ids"], enc_te["attention_mask"])
test_loader = DataLoader(test_ds, batch_size=32)
test_pred = predict_reg(model, test_loader)

submission = pd.DataFrame({"id": test["id"], "target": test_pred})
submission.to_csv("submission_commonlit.csv", index=False)
print("submission_commonlit.csv 저장 완료:", submission.shape)
print(submission.round(4).to_string(index=False))
# 예측이 train 범위(-3.7~1.7) 안이면 ok


# ## 마무리 회고
# 
# | 단계 | 모델 | Validation RMSE |
# |---|---|---|
# | Baseline | TF-IDF + Ridge | 0.698 |
# | 추가 실험 | TF-IDF + 가독성 피처 | 0.693 (-0.009) |
# | **개선** | **DistilBERT (회귀 헤드)** | **0.530** |
# 
# **배운 것 / 느낀 것**
# - 회귀라 평가/진단 도구를 통째로 바꿔야 했다: F1→RMSE, 혼동행렬→예측-실제 산점도, FP/FN→과대·과소예측 사례.
# - 고전적 가독성 공식은 **해석엔 훌륭**했지만(상관 방향이 전부 상식과 일치), TF-IDF 위에서의 성능 기여는 제한적.
# - 결국 **문맥을 읽는 DistilBERT**가 한계를 밀어냈다. 다만 데이터가 작아 과적합 관리(val 기준 best 저장)가 관건이었다.
# - 오차가 큰 글을 직접 읽어보니, '쉬운 단어로 쓴 어려운 내용' / '어려운 단어로 쓴 쉬운 내용'에서 모델이 헷갈린다는 게 보였다.
