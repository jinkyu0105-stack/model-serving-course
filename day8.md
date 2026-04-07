# 한국어 감성 분석기 구현
> 허깅페이스에서 처음으로 모델을 골라보고 배운것을 토대로 사용법을 익히려고 했습니다.
> 태스크는 text-classification 이고 bert-base-multilingual-cased을 Fine-tuning 한 모델입니다.
> 한국어 고객 리뷰를 기반으로 한국어 감성 분석을 위해 정교하게 조정된 모델입니다.

## 간단한 테스트
> 모델 소개 페이지에 간단하게 테스트 해 볼 수 있는 코드가 있었습니다.
```
from transformers import pipeline

sentiment_model = pipeline(model="WhitePeak/bert-base-cased-Korean-sentiment")
sentiment_mode("매우 좋아")
```
> 결과
```
LABEL_0: negative    # "부정" 으로 변경
LABEL_1: positive    # "긍정" 으로 변경함
```

## 버전 문제
> 테스트 코드가 에러가 났으며 파이토치 버전 2.6이하에서 나는 문제였습니다. 