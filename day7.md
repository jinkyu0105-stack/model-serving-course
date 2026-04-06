## 테스트
### 테스트 1
> <img width="600" height="698" alt="image" src="https://github.com/user-attachments/assets/e6cbc9fb-fa69-4a40-aaca-17ce3549dceb" />
---
### 테스트 2
> <img width="607" height="829" alt="image" src="https://github.com/user-attachments/assets/9a2747eb-4851-4bbd-ad83-1513bb96e432" />
---
### 테스트 3
> <img width="783" height="401" alt="image" src="https://github.com/user-attachments/assets/684a902b-61dc-42c4-925c-da38d58406ff" />
---
### 테스트 4
> <img width="933" height="634" alt="image" src="https://github.com/user-attachments/assets/593847c9-1b7e-45fd-abe3-816b2661bb07" />
---


## UI 테스트
### [테스트 A] 기본 대화
<img width="943" height="570" alt="image" src="https://github.com/user-attachments/assets/483cfe0e-778d-4785-af35-b61ed981ee58" />

### [테스트 B] 설정 변경
  1. Temperature를 0.1로 낮추기 → 보수적인 응답 확인
    - <img width="927" height="201" alt="image" src="https://github.com/user-attachments/assets/de565143-a264-47ae-9d04-92f9eb4ec308" />

  3. Temperature를 1.5로 올리기 → 다양한 응답 확인
    - <img width="964" height="232" alt="image" src="https://github.com/user-attachments/assets/83bf0b92-ec5c-41ab-88c4-f1cf96deca6e" />


### [테스트 C] 대화 초기화
  1. 여러 턴 대화 후 "대화 초기화" 클릭
  2. 대화 기록이 사라지는지 확인
     - <img width="1030" height="461" alt="image" src="https://github.com/user-attachments/assets/5614512c-c85b-43d9-a380-28e7b5491e77" />


### [테스트 D] 인증 실패
  1. API Key를 wrong-key로 변경
  2. 메시지 입력 → "🔑 인증 실패" 메시지 확인
     - <img width="443" height="282" alt="image" src="https://github.com/user-attachments/assets/9e185bb5-f3f7-4cb5-9351-1a57659171c8" />



### ✅ Day 7 최종 체크포인트


Q1. Day 5(정형 데이터)와 Day 7(텍스트 생성)에서 전처리 방식의 차이는?

    - 정규화 : 토크나이징
    
Q2. 멀티턴 대화에서 서버가 상태를 유지하지 않는 이유는?

    - 상태를 유지한다는 것은 모든 기록을 일일이 저장 한다는 것이고, 이것은 많은 자원을 필요로 합니다. 
    이것을 무상태성 REST 방식으로 해결하여 서버 자원을 효율적으로 사용할 수 있습니다.
    
Q3. API Key가 잘못되면 서버는 어떤 상태 코드를 반환하고, UI는 어떻게 처리합니까?

    -  401: 채팅 화면에 키가 잘못 되었음을 알려주고, 시스템이 완전히 오류로 사용 못하는 상황이 발생하지 않도록 합니다.
    
Q4. temperature를 낮추면 생성 결과에 어떤 변화가 있습니까?

    - 보수적이고 일관된 답변이 생성됩니다.
    
Q5. 이 서비스를 다른 컴퓨터에서 실행하려면 무엇이 필요합니까?

    - requiresment.txt , 모델 파일 (~/.cacht/Hugginngface/hub) 

---

### 회고
> 모델의 생성 품질은 맘에 들지는 않았지만, 작은 모델이라도 허깅페이스에서 쉽게 다운받아 사용하는 것이 신기했습니다.
> 대략적으로 큰 그림이 이제 그려지는데 오늘 코드를 좀 더 연습하고 공부해서 하이퍼파라미터 변경 또는 모델 교체도 해보려고 합니다.
