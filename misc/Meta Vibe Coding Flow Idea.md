# Meta Vibe Coding Flow Idea

## 요약

- Ideation 단계부터 LLM과 토론해 프로젝트의 방향성 설정

- 프로젝트 단계별로 특화된 Agent 사용

- Agent Prompt를 만드는 Meta Agent 사용

- 전문가별 Resource 제공 시 이를 키워드 중심으로 요약 정리하는 Agent가 보조

- LLM과 토론해 산출한 중간 단계 output을 저장해 다음 단계의 context로 사용

- 가능하다면 LLM과 토론하는데 사용한 prompt와 대화내용도 프로젝트 repo에 포함

## 단계별 특화 Agent 생성

- 단계별 에이전트 rule 사용
  
  1. Ideation과 문제 정의 (목표 설정)
  
  2. 프로젝트 요건 구체화 (스콥, 완성의 정의, RFP)
  
  3. 프로젝트 구조 설계, 일정 산출
  
  4. 개발
  
  5. 테스트 (QA)
  
  6. 배포

- 각 에이전트에게 전문성과 독자적인 view를 줄 수 있는 자료 모으고 요약 Agent 가 키워드 중심 요약

  - 주요 이론

  - 주요 관점 (백과사전 참고)

  - 주요 public figure

  - 주요 논문이나 책, 교안

- Meta Agent 가 특화 에이전트에 맞는 Prompt 생성

## 단계별 세부 단계 설정

- 예를들어 Ideation 단계에서 
  
  - 디자인씽킹 방법론 적용 가능
  
  - TRIZ 방법론도 가능

- 컨설팅 스타일 비판 에이전트도 가능 (reflection 전용)

## 모든 단계의 커뮤니케이션 과정 문서화

- LLM과 토론해 최종 산출물 반환 및 저장

- LLM과 토론한 내용 자체도 그대로/요약하여 repo에 저장

- 프로젝트의 현재 상황, snapshot에 대한 설명도 changelog 처럼 남기고 + README를 단계적으로 업데이트
