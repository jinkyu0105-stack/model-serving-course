
import streamlit as st
import requests

st.set_page_config(
    page_title="한국어 감성 분석기",
    page_icon="😊",
    layout="centered",
)

API_BASE = "http://localhost:8000"

def call_api(url, json_data=None, method="post"):
    try:
        if method == "get":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=json_data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(" 서버에 연결할 수 없습니다. FastAPI 서버가 실행중인지 확인하세요")
        return None
    except requests.exceptions.Timeout:
        st.warning(" 응답 시간 초과. 잠시 후 다시 시도하세요")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f" 서버 에러 (HTTP {e.response.status_code})")
    except Exception as e:
        st.error(f" 오류: {type(e).__name__}")
        return None

st.title("😊 한국어 감성 분석 판독")
st.write("문장을 입력하면 긍정인지 부정인지 분석해 드립니다.")

with st.sidebar:
    st.header("🔐API 보안 설정")
    api_key = st.text_input("API Key", type="password", value="test-key-001")
    st.divider()

    health = call_api(f"{API_BASE}/health", method="get")
    if health and "healthy" in health.get("status", ""):
        st.success("🟢 서버 연결됨")
        server_ok = True
    else:
        st.error("🔴 서버 연결 실패")
        server_ok = False

    st.divider()
    st.caption("Sentiment Predictor Dashboard v1.0")

with st.container():
    user_input = st.text_area(
        "분석할 문장을 입력하세요",
        placeholder="예: 오늘 날씨도 좋고 공부도 잘 돼서 정말 행복해!",
        height=150
    )

if st.button("감성 분석 시작", use_container_width=True):
    if not user_input.strip():
        st.warning("문장을 입력해 주세요.")
    else:
        with st.spinner("AI가 문장을 읽고 분석 중..."):
            try:
                response = requests.post(
                    f"{API_BASE}/predict",
                    json={"text": user_input},
                    headers={"X-API-Key": api_key},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    st.divider()
                    st.subheader("📊분석 결과")

                    label = result["label"]
                    confidence = result["confidence"] * 100

                    col1, col2 = st.columns(2)

                    with col1:
                        if label == "긍정":
                            st.success(f"### 결과: {label} 😊")
                        else:
                            st.error(f"### 결과: {label} 😢")

                    with col2:
                        st.metric("확신도", f"{confidence:.2f}%")
                        st.progress(result["confidence"])

                elif response.status_code == 401:
                    st.error("API Key가 올바르지 않습니다.")
                elif response.status_code == 503:
                    st.warning("모델이 로드 중입니다. 잠시 후 다시 시도해 주세요.")
                else:
                    st.error(f" 에러 발생: {response.json().get('detail', '알 수 없는 에러')}")

            except requests.exceptions.ConnectionError:
                st.error("서버가 실행 중이 아닙니다. 포트 8000을 확인하세요")
            except Exception as e:
                st.error(f"예상치 못한 오류: {str(e)}")

st.divider()
st.caption("Model: WhitePeak/bert-base-cased-Korean-sentiment")
