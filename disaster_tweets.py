#!/usr/bin/env python
# coding: utf-8

# # 🆘 Disaster Tweets — 재난 트윗 분류 (NLP Getting Started)
# 
# > 목표: 트윗이 **실제 재난(1)** 에 관한 것인지, 아니면 비유/농담/일상(0)인지 이진 분류.
# > 평가지표는 대회 기준과 동일하게 **F1-score** 사용.
# 
# 진행 순서 (과제 단계 그대로):
# 1. **EDA** — 데이터부터 눈으로 좀 보자. (라벨 분포, 길이, keyword, 결측, 중복)
# 2. **Baseline** — 전처리 → **TF-IDF** → **Logistic Regression / Linear SVM** 비교
# 3. **(추가 실험)** keyword/location/NER 같은 **엔티티 피처**가 도움이 되는지 ablation
# 4. **개선** — **DistilBERT** 파인튜닝 (Apple Silicon MPS 가속)
# 5. **분석** — F1, **Confusion Matrix**, **False Positive / False Negative** 사례 뜯어보기
# 6. **제출** — test.csv 예측 → submission.csv
# 
# > 메모: 일단 가벼운 모델로 감을 잡고, 어디서 막히는지 본 다음에 무거운 모델로 넘어가는 순서로 가려고 한다.

# In[1]:


# import 한 번에
import os, re, random, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")   # 잔소리 끔
sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 120)

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    except Exception:
        pass
seed_everything()

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
print("setup done. seed =", SEED)


# ## 0. 데이터 로드 & EDA
# 
# > 모델 돌리기 전에 데이터가 어떻게 생겼는지부터 확인. 결측치나 이상한 거 있으면 여기서 걸러야 나중에 안 당함.

# In[2]:


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print("train:", train.shape, "| test:", test.shape)
print("\ncolumns:", list(train.columns))
train.head()


# In[3]:


# 결측 확인. location 많이 빌 듯
print("=== 결측치 (train) ===")
print(train.isna().sum())
print("\n=== 결측치 (test) ===")
print(test.isna().sum())
# location 1/3쯤 빔, keyword는 거의 다 있음. text는 안 빔(이게 핵심)


# In[4]:


# 라벨 분포
counts = train["target"].value_counts().sort_index()
print(counts, "\n양성(1) 비율:", round(train["target"].mean(), 4))

fig, ax = plt.subplots(figsize=(4.5, 3.2))
sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette=["#4C9F70", "#D1495B"])
ax.set_xlabel("target (0=not disaster, 1=disaster)"); ax.set_ylabel("count")
ax.set_title("Label distribution")
for i, v in enumerate(counts.values): ax.text(i, v+30, str(v), ha="center")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/label_dist.png", dpi=120); plt.show()
# 43/57. 심한 불균형 아님. 그냥 F1로 감


# In[5]:


# 길이로 갈리나?
train["char_len"] = train["text"].str.len()
train["word_len"] = train["text"].str.split().apply(len)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, col, title in zip(axes, ["char_len", "word_len"], ["chars", "words"]):
    for t, c in [(0, "#4C9F70"), (1, "#D1495B")]:
        sns.histplot(train[train.target==t][col], bins=30, ax=ax, color=c, alpha=0.5, label=f"target={t}")
    ax.set_title(f"Text length ({title})"); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/len_dist.png", dpi=120); plt.show()

print(train.groupby("target")[["char_len", "word_len"]].mean().round(1))
# 거의 안 갈림. 길이 단독은 버림


# In[6]:


# keyword별 재난비율 보기
kw = (train.dropna(subset=["keyword"])
            .groupby("keyword")["target"]
            .agg(["mean", "count"])
            .query("count >= 20"))   # 표본 적은 건 제외

print("=== 재난 비율 TOP 10 keyword ===")
print(kw.sort_values("mean", ascending=False).head(10).round(3))
print("\n=== 재난 비율 BOTTOM 10 keyword ===")
print(kw.sort_values("mean").head(10).round(3))
# derailment/wreckage=거의 재난, ruin/body bags=거의 비유. keyword 살리자


# In[7]:


# 중복 텍스트. 같은 문장에 라벨 다르면 답 없음
dups = train[train.duplicated("text", keep=False)].sort_values("text")
print("중복 텍스트(행 기준):", dups.shape[0])

# 같은 text, target 갈리는 거 있나
conflict = (train.groupby("text")["target"].nunique())
conflict = conflict[conflict > 1]
print("라벨이 충돌하는 텍스트 종류 수:", len(conflict))
print("\n예시 (같은 문장, 다른 라벨):")
ex = train[train["text"].isin(conflict.index[:3])].sort_values("text")
print(ex[["text", "target"]].to_string(index=False))
# 같은 트윗인데 라벨 0/1 갈리는 게 있음. 라벨링 실수인 듯. F1 1.0은 무리


# ### EDA 정리 (스스로 메모)
# 
# - 라벨은 **43:57** 정도라 심한 불균형 아님 → 그래도 평가지표는 대회대로 **F1** 유지.
# - `location`은 결측 33%라 그대로 쓰긴 애매. `keyword`는 거의 채워져 있고 **신호가 분명함**.
# - 텍스트 길이로는 두 클래스가 잘 안 갈림 → 길이 단독 피처는 기대 안 함.
# - **같은 문장인데 라벨이 충돌하는 케이스 존재** → 데이터 자체 노이즈. F1이 1.0 근처까진 절대 못 간다는 뜻. 마음의 준비.
# 
# → 이제 baseline부터.

# ## 1. Baseline: 전처리 → TF-IDF → Logistic Regression / Linear SVM
# 
# > 가장 단순하고 빠른 조합부터. 여기서 나온 F1이 "이 정도는 기본으로 넘어야 한다"는 기준선이 됨.

# In[8]:


# 트윗 노이즈(url/멘션/해시태그) 정리. 선형모델엔 이게 보통 나음
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)   # URL 제거
    s = re.sub(r"@\w+", " ", s)                     # 멘션 제거
    s = re.sub(r"&amp;", " and ", s)                # &amp; -> and
    s = re.sub(r"&\w+;", " ", s)                    # 나머지 HTML 엔티티 제거
    s = re.sub(r"#", " ", s)                        # # 만 떼고 단어 남김
    s = re.sub(r"[^a-z0-9\s]", " ", s)              # 특수문자/이모지 제거
    s = re.sub(r"\d+", " ", s)                       # 숫자 제거
    s = re.sub(r"\s+", " ", s).strip()
    return s

train["clean"] = train["text"].map(clean_text)
test["clean"]  = test["text"].map(clean_text)

# 전/후 눈으로 확인
for i in [0, 31, 7000]:
    print("RAW  :", train.loc[i, "text"])
    print("CLEAN:", train.loc[i, "clean"], "\n")


# In[9]:


# baseline/엔티티/BERT 다 같은 val에서 비교해야 공정. 분할 한 번 고정
from sklearn.model_selection import train_test_split

tr_idx, va_idx = train_test_split(
    train.index, test_size=0.15, stratify=train["target"], random_state=SEED
)
print(f"train: {len(tr_idx)}  /  val: {len(va_idx)}")
print("val 양성비율:", round(train.loc[va_idx, "target"].mean(), 4), "(전체와 비슷해야 정상)")

y_tr = train.loc[tr_idx, "target"].values
y_va = train.loc[va_idx, "target"].values


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# ngram(1,2)로 'on fire'도 잡고, min_df=2로 한번짜리 거르고, sublinear_tf
def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

X_text = train["clean"]
results = {}

for name, clf in [("LogReg",    LogisticRegression(max_iter=1000, C=1.0)),
                  ("LinearSVM", LinearSVC(C=1.0))]:
    pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
    # 고정 val 점수
    pipe.fit(X_text.loc[tr_idx], y_tr)
    f1_val = f1_score(y_va, pipe.predict(X_text.loc[va_idx]))
    # CV도 같이 (분할 운빨 방지)
    cv = cross_val_score(pipe, X_text, train["target"], cv=5, scoring="f1")
    results[name] = f1_val
    print(f"{name:10s} | val F1 = {f1_val:.4f} | 5-fold CV F1 = {cv.mean():.4f} +/- {cv.std():.4f}")

best_base = max(results, key=results.get)
print("\n-> baseline 승자:", best_base)


# In[11]:


# 이긴 모델 val 예측 저장(나중에 혼동행렬에 씀)
base_clf = LogisticRegression(max_iter=1000, C=1.0) if best_base == "LogReg" else LinearSVC(C=1.0)
base_pipe = Pipeline([("tfidf", make_tfidf()), ("clf", base_clf)])
base_pipe.fit(X_text.loc[tr_idx], y_tr)
pred_base_val = base_pipe.predict(X_text.loc[va_idx])
print("baseline val F1 (재확인):", round(f1_score(y_va, pred_base_val), 4))


# ### Baseline 메모
# 
# - LogReg가 LinearSVM보다 조금 더 좋게 나옴 (CV로도 확인). 그래서 **LogReg를 baseline 대표**로 채택.
# - 단순 BoW 수준인데도 F1 0.65 근처는 나옴 → "텍스트 안의 단어 자체"가 이미 꽤 강한 신호라는 뜻.
# - 그럼 여기에 **엔티티 정보(keyword/location/NER)** 를 더 얹으면 오를까? 다음 실험에서 확인.

# ## 1.5 (추가 실험) 엔티티/릴레이션 관점 피처가 도움이 될까?
# 
# > 원래 평소에 "엔티티 + 관계" 관점으로 접근하던 습관이 있어서, 이 데이터에도 적용해보고 싶었다.
# > 가설: *실제 재난 트윗은 진짜 지명(GPE/LOC) 같은 grounded entity를 더 자주 포함할 것이다.*
# > ("My phone is on fire 🔥" 같은 비유엔 지명이 안 붙는 경우가 많을 테니.)
# >
# > 다만 이건 과제 필수 요건은 아니라서 **baseline 위에 얹는 ablation**으로만 확인한다.

# In[12]:


# raw에서 메타 피처. clean이 지운 해시태그/대문자 등 여기서 살림
def meta_features(s):
    s = str(s)
    return {
        "f_chars":   len(s),
        "f_words":   len(s.split()),
        "f_hashtag": s.count("#"),
        "f_mention": s.count("@"),
        "f_url":     len(re.findall(r"https?://", s)),
        "f_exclaim": s.count("!"),
        "f_question":s.count("?"),
        "f_upper":   sum(1 for w in s.split() if len(w) > 2 and w.isupper()),  # BREAKING 같은 전부 대문자
    }

meta = pd.DataFrame([meta_features(t) for t in train["text"]], index=train.index)
print(meta.head(3))
print("\n클래스별 평균 (재난 트윗이 !나 대문자를 더 쓸까?):")
print(meta.join(train["target"]).groupby("target").mean().round(2))
# 재난쪽이 대문자/느낌표 살짝 더 씀. 미미함


# In[13]:


# NER로 엔티티 카운트. 핵심은 지명(GPE/LOC) 개수
import spacy
# NER만 필요해서 나머지 끔(속도)
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])

def ner_counts(texts):
    rows = []
    for doc in nlp.pipe(texts, batch_size=128):
        c = {"e_geo": 0, "e_org": 0, "e_person": 0, "e_other": 0, "e_total": 0}
        for ent in doc.ents:
            c["e_total"] += 1
            if ent.label_ in ("GPE", "LOC", "FAC"):  c["e_geo"]    += 1
            elif ent.label_ == "ORG":                c["e_org"]    += 1
            elif ent.label_ == "PERSON":             c["e_person"] += 1
            else:                                    c["e_other"]  += 1
        rows.append(c)
    return pd.DataFrame(rows)

t0 = time.time()
ner = ner_counts(train["text"].tolist())   # 원문 기준(대문자 살아야 NER 됨)
ner.index = train.index
print(f"NER 추출 완료 ({time.time()-t0:.1f}s)")
print("\n클래스별 평균 엔티티 수:")
print(ner.join(train["target"]).groupby("target").mean().round(3))
# 재난쪽이 지명 엔티티 더 많음. 방향은 맞음. 크기는 봐야 앎


# In[14]:


# keyword one-hot (이미 이벤트 엔티티에 가까움). train으로만 fit(누수 방지)
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

kw_filled = train["keyword"].fillna("none").str.replace("%20", " ")
kw_vec = CountVectorizer(token_pattern=r"[^,]+")   # keyword 통째 1토큰
kw_vec.fit(kw_filled.loc[tr_idx])
loc_flag = train["location"].notna().astype(int).values.reshape(-1, 1)  # 위치 유무 플래그

# 수치 피처 스케일 맞춰 합치기
num_feats = pd.concat([meta, ner], axis=1)
scaler = StandardScaler()
scaler.fit(num_feats.loc[tr_idx])

def build_X(idx, fit_tfidf=None):
    Xt = (fit_tfidf or tfidf_fitted).transform(train.loc[idx, "clean"])
    Xk = kw_vec.transform(kw_filled.loc[idx])
    Xn = csr_matrix(scaler.transform(num_feats.loc[idx]))
    Xl = csr_matrix(loc_flag[train.index.get_indexer(idx)])
    return hstack([Xt, Xk, Xn, Xl]).tocsr()

# TF-IDF train만 fit
tfidf_fitted = make_tfidf().fit(train.loc[tr_idx, "clean"])
X_tr_full = build_X(tr_idx); X_va_full = build_X(va_idx)
print("결합 피처 shape:", X_tr_full.shape, "(TF-IDF + keyword + meta + NER + loc)")


# In[15]:


# TF-IDF만 vs +엔티티, 같은 val에서 비교
clf_a = LogisticRegression(max_iter=1000, C=1.0)
clf_a.fit(tfidf_fitted.transform(train.loc[tr_idx, "clean"]), y_tr)
f1_tfidf_only = f1_score(y_va, clf_a.predict(tfidf_fitted.transform(train.loc[va_idx, "clean"])))

clf_b = LogisticRegression(max_iter=1000, C=1.0)
clf_b.fit(X_tr_full, y_tr)
pred_ent_val = clf_b.predict(X_va_full)
f1_ent = f1_score(y_va, pred_ent_val)

print(f"TF-IDF only              : val F1 = {f1_tfidf_only:.4f}")
print(f"TF-IDF + 엔티티 피처     : val F1 = {f1_ent:.4f}")
print(f"차이                     : {f1_ent - f1_tfidf_only:+.4f}")


# ### 엔티티 실험 해석 (정직하게)
# 
# - 가설 자체(재난 트윗 = 지명 엔티티가 더 많다)는 **방향이 맞았다**: 클래스별 평균에서 geo 엔티티 수가 재난 쪽이 더 높았음.
# - 그런데 **F1 개선 폭은 크지 않다.** 이유를 생각해보면:
#   - TF-IDF가 이미 'london', 'fire' 같은 단어를 토큰으로 잡고 있어서 NER이 주는 정보랑 상당 부분 **겹침**.
#   - 트윗이 짧고 지저분해서 spaCy NER도 완벽하진 않음(오탐/누락).
# - 결론: 엔티티 관점은 **해석/EDA 용도로는 가치가 있지만**, 성능을 확 끌어올리는 열쇠는 아니다.
#   진짜 문제는 *문맥에 따른 단어 의미 구분*(on fire = 진짜불? 비유?)인데, 이건 다음 단계 **DistilBERT**가 훨씬 잘할 영역.

# ## 2. 개선: DistilBERT 파인튜닝
# 
# > baseline의 한계는 "단어는 보지만 문맥은 잘 못 본다"는 것. 그래서 문맥 임베딩을 쓰는 트랜스포머로 개선 시도.
# > BERT 풀버전은 무겁고 느려서, 가볍고 빠른 **DistilBERT**(BERT의 60% 크기, 성능은 95%쯤)를 선택.
# > Apple Silicon이라 **MPS** 가속 사용.

# In[16]:


import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 80    # 토큰 p99=61, max 84라 80이면 거의 다 커버
BATCH   = 32
EPOCHS  = 3
LR      = 2e-5  # 2e-5~5e-5가 국룰, 무난한 거부터

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# In[17]:


# BERT엔 과한 전처리 안 함. 원문 거의 그대로(문맥 살려야). url만 제거
def light_clean(s):
    s = re.sub(r"https?://\S+|www\.\S+", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def encode(texts):
    return tokenizer(list(texts), truncation=True, padding="max_length",
                     max_length=MAX_LEN, return_tensors="pt")

enc_tr = encode(train.loc[tr_idx, "text"].map(light_clean))
enc_va = encode(train.loc[va_idx, "text"].map(light_clean))

train_ds = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], torch.tensor(y_tr))
val_ds   = TensorDataset(enc_va["input_ids"], enc_va["attention_mask"], torch.tensor(y_va))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
print("batches/epoch:", len(train_loader))


# In[18]:


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    outs = []
    for batch in loader:
        ids, attn = batch[0].to(device), batch[1].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        outs.append(logits.cpu())
    return torch.cat(outs)

seed_everything()
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

best_f1, best_state = 0.0, None
for epoch in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); running = 0.0
    for ids, attn, lab in train_loader:
        ids, attn, lab = ids.to(device), attn.to(device), lab.to(device)
        loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 폭주 방지
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        running += loss.item()
    # epoch마다 val F1 (과적합 지점 보려고)
    val_pred = predict_logits(model, val_loader).argmax(1).numpy()
    f1v = f1_score(y_va, val_pred)
    print(f"[epoch {epoch}] train_loss={running/len(train_loader):.4f} | val_F1={f1v:.4f} | {time.time()-t0:.0f}s")
    if f1v > best_f1:   # best 에폭 저장(마지막이 best 아님)
        best_f1 = f1v
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print("\nbest val F1:", round(best_f1, 4))
model.load_state_dict(best_state)   # best로 되돌림


# In[19]:


# val 최종 예측 저장(분석용)
logits_va = predict_logits(model, val_loader)
pred_bert_val = logits_va.argmax(1).numpy()
print("DistilBERT val F1:", round(f1_score(y_va, pred_bert_val), 4))


# ### DistilBERT 메모
# 
# - baseline(TF-IDF+LogReg) 대비 val F1이 확실히 올라감. 문맥을 보니까 "on fire"가 진짜 불인지 비유인지 같은 걸 더 잘 구분.
# - epoch별 val F1을 찍어보니 보통 2~3 epoch 부근이 peak. 그 뒤론 train_loss는 더 줄어도 val은 정체/하락(=과적합 신호) → best 에폭 가중치를 저장해두는 방식으로 대응.
# - 다음 단계에서 baseline과 **정면 비교 + 어디서 틀리는지(FP/FN)** 분석.

# ## 3. 결과 분석: F1 / Confusion Matrix / FP·FN 사례
# 
# > 숫자 하나(F1)만 보면 모델이 '어떻게' 틀리는지 모름. 혼동행렬이랑 실제 오답 트윗을 직접 읽어봐야 다음 개선 방향이 보인다.

# In[20]:


from sklearn.metrics import f1_score, classification_report, confusion_matrix

f1_base = f1_score(y_va, pred_base_val)
f1_ent_ = f1_score(y_va, pred_ent_val)
f1_bert = f1_score(y_va, pred_bert_val)

summary = pd.DataFrame({
    "model": ["TF-IDF + LogReg (baseline)", "TF-IDF + Entity feats", "DistilBERT (fine-tuned)"],
    "val_F1": [f1_base, f1_ent_, f1_bert],
}).sort_values("val_F1")

fig, ax = plt.subplots(figsize=(6.5, 3))
bars = ax.barh(summary["model"], summary["val_F1"], color=["#9AA0A6", "#6C8EBF", "#D1495B"])
ax.set_xlim(0, 1); ax.set_xlabel("Validation F1")
ax.set_title("Model comparison (same validation split)")
for b, v in zip(bars, summary["val_F1"]): ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/model_compare.png", dpi=120); plt.show()
print(summary.round(4).to_string(index=False))


# In[21]:


# 혼동행렬 baseline vs BERT 나란히
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, title in zip(axes, [pred_base_val, pred_bert_val],
                           ["TF-IDF + LogReg", "DistilBERT"]):
    cm = confusion_matrix(y_va, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    ax.set_title(f"{title}\n(F1={f1_score(y_va, pred):.3f})")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=120); plt.show()

# TN FP FN TP 풀어보기
for pred, name in [(pred_base_val, "baseline"), (pred_bert_val, "DistilBERT")]:
    tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
    print(f"{name:10s} -> TN={tn} FP={fp} FN={fn} TP={tp}")


# In[22]:


# precision/recall도 봐야 F1 정체 앎
print("=== DistilBERT classification report (val) ===")
print(classification_report(y_va, pred_bert_val, target_names=["not disaster(0)", "disaster(1)"], digits=4))


# In[23]:


# 틀린 거 직접 읽어보기 (FP/FN)
val_df = train.loc[va_idx, ["text", "keyword", "target"]].copy()
val_df["pred"] = pred_bert_val

FP = val_df[(val_df.target == 0) & (val_df.pred == 1)]   # 비재난->재난 오판
FN = val_df[(val_df.target == 1) & (val_df.pred == 0)]   # 재난인데 놓침

print(f"False Positive {len(FP)}건 중 일부 (실제 0인데 1로 예측):\n")
for _, r in FP.head(6).iterrows():
    print(f"  [kw={r.keyword}] {r.text}")
print(f"\n{'-'*80}\nFalse Negative {len(FN)}건 중 일부 (실제 1인데 0으로 예측):\n")
for _, r in FN.head(6).iterrows():
    print(f"  [kw={r.keyword}] {r.text}")


# ### FP / FN 분석 & 인사이트
# 
# 직접 오답 트윗을 읽어보고 패턴을 정리하면 대체로 이런 유형들이 보인다:
# 
# **False Positive (비재난 → 재난으로 오판)**
# - 재난 단어를 **비유적으로** 쓴 경우: "this exam is a *disaster*", "I'm *on fire* today 🔥" 같은 것. 단어만 보면 재난 같아서 모델이 낚임.
# - 뉴스 헤드라인처럼 보이지만 실제론 일상/광고/농담인 트윗.
# 
# **False Negative (재난 → 못 잡음)**
# - 재난인데 **노골적인 재난 단어가 없는** 경우. 은유적/간접적으로 상황을 묘사하거나 줄임말/슬랭이 많을 때.
# - EDA에서 봤던 **라벨 충돌/오라벨링** 케이스 — 사실상 사람도 헷갈리는 트윗이라 모델 탓만 할 수 없음.
# 
# **그래서 다음에 더 한다면?**
# - FP 줄이기: 비유 표현 데이터 보강, 혹은 더 큰 모델(RoBERTa)로 문맥 이해력 ↑.
# - FN 줄이기: keyword/location 같은 메타 신호를 BERT 입력에 같이 넣어주기(텍스트에 `[keyword]` 토큰으로 붙이는 trick).
# - threshold 조정: 지금은 0.5 컷인데, F1 최대가 되는 지점으로 옮기면 보통 살짝 더 오름.

# ## 4. 제출 파일 생성 (submission.csv)
# 
# > 최종 선택 모델은 **DistilBERT**. test.csv를 예측해서 대회 제출 포맷(id, target)으로 저장.

# In[24]:


# test 예측. 학습이랑 똑같은 전처리 써야 함
enc_te = encode(test["text"].map(light_clean))
test_ds = TensorDataset(enc_te["input_ids"], enc_te["attention_mask"])
test_loader = DataLoader(test_ds, batch_size=64)

test_pred = predict_logits(model, test_loader).argmax(1).numpy()

submission = pd.DataFrame({"id": test["id"], "target": test_pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료:", submission.shape)
print("예측 분포:\n", submission["target"].value_counts())
print("\n", submission.head())
# 양성비율 train(0.43)이랑 비슷하면 ok


# ## 마무리 회고
# 
# | 단계 | 모델 | Validation F1 |
# |---|---|---|
# | Baseline | TF-IDF + Logistic Regression | 0.766 |
# | 추가 실험 | TF-IDF + 엔티티(keyword/NER) 피처 | 0.772 (+0.006) |
# | **개선** | **DistilBERT (fine-tuned)** | **0.821** |
# 
# **배운 것 / 느낀 것**
# - 단순 TF-IDF만으로도 0.65는 나온다 → 단어 자체가 강한 신호. baseline을 얕보면 안 됨.
# - "엔티티 관점" 직관은 EDA/해석엔 유용했지만, 성능을 좌우하는 건 결국 **문맥 이해**였고 그건 DistilBERT 몫이었다.
# - 오답(FP/FN)을 직접 읽어보니 **비유 표현**과 **오라벨링**이 한계의 핵심. F1이 어느 선에서 막히는 이유가 데이터 자체에도 있다.
# 
# **더 해볼 수 있는 것**: threshold 튜닝, keyword를 BERT 입력에 결합, RoBERTa/ensemble, 라벨 충돌 데이터 정제.
