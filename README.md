# Korea Lotto AI Predictor v4.1

Streamlit + VS Code + Hugging Face Router 기반 한국 로또 통계/백테스트/LLM 분석 앱입니다.

## v4.1 수정 사항

- Streamlit 버튼 rerun 문제 수정: 예측 결과를 `st.session_state`에 저장하여 `HF LLM으로 추가 분석` 버튼이 정상 작동합니다.
- `HF_MODEL_ID` 별칭 지원: Secrets에 `HF_MODEL_ID`만 넣어도 모델 후보에 반영됩니다.
- HF Router 연결 테스트 버튼 추가
- 모델 후보/에러 표시 개선
- `stream=false` 명시 및 HTTP 오류 메시지 확대

## Streamlit Cloud Secrets 예시

```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_XXXXXXXXXX"
HF_MODEL_ID = "google/gemma-4-26B-A4B-it"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = "1400"
HF_TEMPERATURE = "0.2"
HF_TIMEOUT_CONNECT = "10"
HF_TIMEOUT_READ = "120"
HF_MAX_RETRIES = "3"
```

## 로컬 실행

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
streamlit run app.py
```

## 주의

- 실제 `HF_TOKEN`은 GitHub에 올리지 마세요.
- 로또는 무작위 확률 게임이며, 본 앱은 통계/엔터테인먼트 목적의 후보 생성 도구입니다. 당첨을 보장하지 않습니다.

---

# Korea Lotto AI Predictor v4

한국 로또 6/45 CSV 데이터를 기반으로 **롤링 백테스트 → 전략별 가중치 최적화 → 후보 조합 생성 → Hugging Face Router LLM 추가 분석**까지 실행하는 VS Code / Streamlit 배포형 프로젝트입니다.

> 주의: 로또는 무작위 확률 게임입니다. 이 앱은 과거 데이터 기반 통계 후보 생성 도구이며 당첨이나 수익을 보장하지 않습니다.

---

## 1. 프로젝트 구조

```text
korea_lotto_ai_streamlit_v4/
├─ app.py                         # Streamlit 웹앱
├─ cli.py                         # VS Code/터미널 실행용 CLI
├─ requirements.txt
├─ .env.example                   # 로컬 환경변수 예시
├─ .gitignore
├─ .streamlit/
│  └─ secrets.toml.example        # Streamlit Cloud Secrets 예시
├─ data/
│  └─ .gitkeep                    # CSV를 여기에 넣을 수 있음
└─ src/
   ├─ config.py                   # 환경변수/Secrets 로더
   ├─ lotto_engine.py             # 백테스트/전략/예측 엔진
   └─ llm_engine.py               # HF Router LLM 분석 엔진
```

---

## 2. CSV 형식

권장 컬럼명:

```text
회차, 추첨일, 당첨번호1, 당첨번호2, 당첨번호3, 당첨번호4, 당첨번호5, 당첨번호6, 보너스
```

또는 아래 형태도 자동 감지합니다.

```text
draw, date, num1, num2, num3, num4, num5, num6, bonus
```

CSV 파일을 `data/korealotto.csv`로 넣으면 기본 설정 그대로 실행됩니다.

---

## 3. VS Code 로컬 실행

### 3-1. 가상환경 생성

Windows PowerShell:

```powershell
cd C:\0MyWork1\korea_lotto_ai_streamlit_v4
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3-2. `.env` 생성

```powershell
copy .env.example .env
notepad .env
```

`.env`에 실제 토큰을 넣습니다.

```env
LLM_ENGINE=hf_api
HF_TOKEN=hf_실제토큰
HF_ROUTER_MODEL=google/gemma-4-26B-A4B-it:deepinfra
HF_MODEL_CANDIDATES=google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together
HF_MAX_TOKENS=1400
HF_TEMPERATURE=0.2
HF_TIMEOUT_CONNECT=10
HF_TIMEOUT_READ=120
HF_MAX_RETRIES=3
```

### 3-3. Streamlit 실행

```powershell
streamlit run app.py
```

브라우저가 열리면 CSV를 업로드하거나 `data/korealotto.csv` 경로를 그대로 사용합니다.

### 3-4. CLI 실행

```powershell
python cli.py --csv data/korealotto.csv
```

LLM 분석까지 실행:

```powershell
python cli.py --csv data/korealotto.csv --llm --note "최근 30회 흐름을 더 중시해서 해석해줘"
```

---

## 4. Streamlit Cloud 배포

1. GitHub 새 저장소를 만듭니다.
2. 이 폴더 전체를 push합니다.
3. Streamlit Cloud에서 `app.py`를 entry point로 지정합니다.
4. App Settings → Secrets에 아래 값을 넣습니다.

```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_실제토큰"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = "1400"
HF_TEMPERATURE = "0.2"
HF_TIMEOUT_CONNECT = "10"
HF_TIMEOUT_READ = "120"
HF_MAX_RETRIES = "3"
```

Streamlit Cloud에는 `.env`를 올리지 마세요. 토큰은 반드시 Secrets에 넣으세요.

---

## 5. 이번 v4의 개선점

- 토요일 당일 실행 시 오늘 날짜를 추첨일로 반환하는 날짜 로직 적용
- 데이터 누수 없는 rolling-forward 백테스트
- 빈도, 갭, 모멘텀, 핫콜드, 페어 공출현, 분포 필터, anti-crowd 전략 통합
- 백테스트 성과 기반 softmax 가중치 자동 최적화
- 후보 조합 점수화: 합계, 홀짝, 고저, 연속수, 직전 회차 반복, 페어 공출현 반영
- Streamlit UI에서 후보 조합, 전략 가중치, 백테스트 표/차트 확인
- Hugging Face Router API 기반 LLM 추가 분석
- 로컬 모델 다운로드 없이 Streamlit Cloud 배포 가능

---

## 6. 오류 해결

### HF_TOKEN이 없다고 나오는 경우

- 로컬: `.env` 파일이 프로젝트 루트에 있는지 확인합니다.
- Cloud: Streamlit Secrets에 `HF_TOKEN`을 넣었는지 확인합니다.
- Streamlit을 이미 실행 중이었다면 재시작합니다.

### 모델이 provider에서 지원되지 않는다는 오류

`HF_MODEL_CANDIDATES`에 여러 후보가 들어 있으므로 앱이 순차적으로 fallback합니다. 그래도 실패하면 Hugging Face 계정에서 해당 모델/프로바이더 접근 권한 또는 결제/Provider 활성화 상태를 확인하세요.

### CSV 컬럼 인식 실패

컬럼명을 아래처럼 맞추면 가장 안정적입니다.

```text
회차, 추첨일, 당첨번호1, 당첨번호2, 당첨번호3, 당첨번호4, 당첨번호5, 당첨번호6, 보너스
```
