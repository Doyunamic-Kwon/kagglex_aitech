# -*- coding: utf-8 -*-
# 마크다운/주석 간결화: 이모지, '목표:', '메모', '고전적', MPS 부연 등 제거. 코드 출력은 보존.
import re, nbformat

# 이모지만 제거 (화살표 → · — 같은 문장부호는 유지)
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F]+",
    flags=re.UNICODE)
def deemoji(s):
    return re.sub(r"[ ]{2,}", " ", EMOJI.sub("", s)).replace("# #", "##")

# (식별 키, 새 본문) — 키가 들어있는 markdown 셀을 통째로 교체
DIS = [
("(NLP Getting Started)",
"""# Disaster Tweets 재난 트윗 분류

트윗이 실제 재난(1)인지 아닌지 분류. 평가지표 F1.

1. EDA
2. Baseline: TF-IDF + LogReg / LinearSVM
3. 엔티티 피처 실험
4. DistilBERT
5. 분석 (F1, Confusion Matrix, FP/FN)
6. 제출"""),
("## 0. 데이터 로드", "## 0. 데이터 로드 & EDA"),
("### EDA 정리",
"""### EDA 정리

- 라벨 43:57, 심한 불균형 아님
- location 결측 33%, keyword는 거의 채워짐
- 길이로는 두 클래스 안 갈림
- 같은 문장에 라벨 충돌 있음 → F1 1.0은 불가능"""),
("## 1. Baseline: 전처리 → TF-IDF → Logistic", "## 1. Baseline: TF-IDF + LogReg / LinearSVM"),
("### Baseline 메모",
"""### Baseline 결과

- LogReg가 LinearSVM보다 나음 → LogReg 채택
- BoW만으로도 F1 0.76
- 다음: 엔티티 피처가 도움 되는지"""),
("## 1.5 (추가 실험) 엔티티",
"""## 1.5 엔티티 피처 실험

가설: 실제 재난 트윗은 지명 같은 엔티티를 더 자주 포함할 것이다. baseline에 얹어서 확인."""),
("### 엔티티 실험 해석",
"""### 엔티티 실험 결과

- 재난 트윗이 지명 엔티티를 더 많이 가짐. 방향은 맞음
- 그래도 F1 개선은 작음. TF-IDF랑 정보가 겹침
- 문맥 구분은 DistilBERT가 할 일"""),
("## 2. 개선: DistilBERT 파인튜닝",
"""## 2. 개선: DistilBERT

baseline은 문맥을 못 봄. DistilBERT로 개선."""),
("### DistilBERT 메모",
"""### DistilBERT 결과

- baseline보다 F1 상승
- 2~3 epoch에서 best, 이후 과적합 → best 가중치 저장"""),
("## 3. 결과 분석: F1", "## 3. 결과 분석: F1 / Confusion Matrix / FP·FN"),
("### FP / FN 분석",
"""### FP / FN 분석

- FP: 재난 단어를 비유로 쓴 경우. on fire 같은
- FN: 재난인데 재난 단어가 없거나 라벨이 잘못된 경우
- 개선: threshold 조정, keyword를 입력에 결합, 더 큰 모델"""),
("## 4. 제출 파일 생성 (submission.csv)", "## 4. 제출 파일 생성"),
("## 마무리 회고",
"""## 마무리

| 단계 | 모델 | Validation F1 |
|---|---|---|
| Baseline | TF-IDF + LogReg | 0.766 |
| 추가 실험 | TF-IDF + 엔티티 피처 | 0.772 |
| 개선 | DistilBERT | 0.821 |

- BoW만으로도 0.76. 엔티티 피처는 효과 작음. 개선은 DistilBERT
- 한계: 비유 표현, 라벨 오류"""),
]

COM = [
("# 📚 CommonLit",
"""# CommonLit Readability 글 난이도 예측 (회귀)

지문(excerpt)의 읽기 난이도 target 예측. 낮을수록 어려움. 평가지표 RMSE.

1. EDA
2. Baseline: TF-IDF + Ridge
3. 가독성 피처 실험
4. DistilBERT 회귀
5. 분석 (RMSE, 예측-실제 산점도, 잔차, 과대/과소예측)
6. 제출

test는 7행뿐. 코드대회라 실제 채점셋은 숨겨져 있음."""),
("### EDA 메모",
"""### EDA 정리

- 쓸 컬럼은 excerpt 하나. url/license는 제외
- target은 종모양 → 변환 없이 회귀
- 길이로는 난이도 안 갈림 → 어휘/구조가 핵심"""),
("## 1. Baseline: 전처리 → TF-IDF → Ridge", "## 1. Baseline: TF-IDF + Ridge"),
("### Baseline 메모",
"""### Baseline 결과

- TF-IDF + Ridge로 RMSE 0.70. naive(평균예측 1.03)보다 좋음
- 선형모델은 어순/문맥을 못 봐서 한계
- 다음: 가독성 피처, 그다음 DistilBERT"""),
("## 1.5 (추가 실험) 고전적",
"""## 1.5 가독성 피처 실험

가설: 문장 길이, 음절 수 같은 가독성 지표를 더하면 도움이 될 것이다. baseline에 얹어서 확인."""),
("### 가독성 실험 해석",
"""### 가독성 실험 결과

- 가독성 지표들은 target과 상식적인 방향의 상관. 방향은 맞음
- 그래도 RMSE 개선은 작음
- 문맥은 DistilBERT가 할 일"""),
("## 2. 개선: DistilBERT 회귀 헤드",
"""## 2. 개선: DistilBERT 회귀

출력 1개(num_labels=1) + MSE로 점수 회귀. 지문이 길어서 max_length=256."""),
("### DistilBERT 메모",
"""### DistilBERT 결과

- baseline보다 RMSE 하락
- 데이터 작아 금방 과적합 → best 시점 저장"""),
("## 3. 결과 분석: RMSE", "## 3. 결과 분석: RMSE / 예측-실제 / 잔차 / 과대·과소예측"),
("### 잔차/오차 분석",
"""### 잔차/오차 분석

- DistilBERT 점들이 대각선에 더 밀착. 개선 확인
- 잔차는 0 중심, DistilBERT가 더 좁음
- 과대예측: 쉬운 단어인데 내용이 어려운 글 / 과소예측: 어려운 단어인데 내용은 단순한 글
- 개선: target 정규화, 앙상블, 더 긴 max_length"""),
("## 4. 제출 파일 생성", "## 4. 제출 파일 생성"),
("## 마무리 회고",
"""## 마무리

| 단계 | 모델 | Validation RMSE |
|---|---|---|
| Baseline | TF-IDF + Ridge | 0.698 |
| 추가 실험 | TF-IDF + 가독성 피처 | 0.693 |
| 개선 | DistilBERT | 0.530 |

- 가독성 피처는 효과 작음. 개선은 DistilBERT
- 오차 큰 글: 쉬운 단어/어려운 내용, 어려운 단어/쉬운 내용에서 헷갈림"""),
]

def run(nb_path, builder_path, repl):
    nb = nbformat.read(nb_path, as_version=4)
    btext = open(builder_path, encoding="utf-8").read()
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for key, new in repl:
                if key in cell.source:
                    old = cell.source
                    if old in btext:
                        btext = btext.replace(old, new)
                    cell.source = new
                    break
        cell.source = deemoji(cell.source)
    nbformat.write(nb, nb_path)
    open(builder_path, "w", encoding="utf-8").write(deemoji(btext))
    # 남은 이모지 점검
    left = sum(len(EMOJI.findall(c.source)) for c in nb.cells)
    print(f"{nb_path}: 남은 이모지 {left}개")

run("disaster_tweets.ipynb", "build_notebook.py", DIS)
run("commonlit_readability.ipynb", "build_notebook_commonlit.py", COM)
print("done")
