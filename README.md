# Kaggle NLP 과제 — 재난 트윗 분류 & 글 난이도 예측

가천대 202037006 권도윤. Kaggle NLP 2개 대회를 단계별(EDA → Baseline → 딥러닝 개선 → 결과 분석 → 제출)로 진행.
한쪽은 **분류**, 다른 쪽은 **회귀**라 평가/분석 도구를 다르게 적용했다.

## 과제1 — Disaster Tweets (이진 분류, F1)
- 노트북 [`disaster_tweets.ipynb`](disaster_tweets.ipynb) · 순수 소스 [`disaster_tweets.py`](disaster_tweets.py)
- 흐름: TF-IDF + LogReg(0.766) → 엔티티 피처 실험(0.772) → **DistilBERT(0.821)**
- **Kaggle Public LB F1 = 0.83573**
- 분석: F1, Confusion Matrix, False Positive / False Negative 사례
- 제출물: `202037006_권도윤_kaggle과제1.zip`, 제출 캡처 [`screenshots/disaster_submission.png`](screenshots/disaster_submission.png)

## 과제2 — CommonLit Readability (회귀, RMSE)
- 노트북 [`commonlit_readability.ipynb`](commonlit_readability.ipynb) · 순수 소스 [`commonlit_readability.py`](commonlit_readability.py)
- 흐름: TF-IDF + Ridge(0.698) → 가독성 피처 실험(0.693) → **DistilBERT 회귀(0.530)**
- **Kaggle LB RMSE = 0.547 / 0.550** (Code Competition — 노트북 제출, 오프라인 모델)
- 분석: RMSE, 예측-실제 산점도, 잔차 분포, 과대/과소예측 사례
- 제출물: `202037006_권도윤_kaggle과제2.zip`, 제출 캡처 [`screenshots/commonlit_submission.png`](screenshots/commonlit_submission.png)

## 구조
```
disaster_tweets.ipynb / .py          # 과제1 (분류)
commonlit_readability.ipynb / .py    # 과제2 (회귀)
build_notebook*.py                   # 노트북 생성 빌더
train_save_commonlit.py              # 커먼릿 DistilBERT 학습·저장 (제출용)
kaggle_kernel/                       # 커먼릿 제출용 추론 커널(Code Competition)
figures/ , figures_commonlit/        # 시각화 PNG
submission*.csv                      # 제출 파일
screenshots/                         # Kaggle 제출 캡처
data/ , data_commonlit/              # 대회 데이터 (git 제외 — 아래로 다운로드)
```

## 실행
```bash
pip install scikit-learn pandas numpy matplotlib seaborn torch transformers spacy nbconvert
python -m spacy download en_core_web_sm
# 데이터 (Kaggle API + ~/.kaggle/kaggle.json, 각 대회 규칙 동의 선행)
kaggle competitions download -c nlp-getting-started -p data && unzip -o data/*.zip -d data
kaggle competitions download -c commonlitreadabilityprize -p data_commonlit && unzip -o data_commonlit/*.zip -d data_commonlit
```

> 대회 데이터는 Kaggle 약관상 재배포 금지라 저장소에 포함하지 않는다.
