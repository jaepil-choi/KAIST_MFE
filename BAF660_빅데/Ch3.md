# Ch3 필기

## Lending Club data intro
- 이미 전처리 다양하게 적용하신 것을 올려줬다고 함. 
- 일단 Y를 다르게 둔 것 같다. charged_off 를 Y로 둠. 

## Train Test Split 할 때의 주의점
- Over/Under Sampling, 교수님은 안하심. 5:1 정도로 비율 양호해서. 
- over/under 할 경우, CV 때 까다로워짐. hyper param tuning을 위해 RandomCV 쓰는데, 이 때 Validation set 에 synthetic 들어가면 안된다고 하심. 

## XGBoost

### Python Wrapper Package vs Scikit-Learn Wrapper Package
- 두 가지로 나눠서 배포됨. scikit learn interface에 맞춰 fit(), predict() 등을 지원함. 
- 파이썬, 매소드/파라미터 name 등이 조금 다를 수 있음. 덜 익숙. 

### XGBoost param 들

#### General params
- gbtree 설명하며, tree 말고도 boosting의 아이디어를 가진 것들이 많다고 함. 
- `n_jobs` 로 병렬 연산. 

#### Booster params
- `learning_rate`: 작을수록, 하나하나의 tree의 영향도를 줄여줌. 
- `min_child_weight`: 자식노드들에 속한 자료들의 weight의 합이 최소한 얼마가 되어야 한다는 것.
    - default = 1. 
        - classification에서는 log loss를 많이 쓰는데, 이 땐 cover = p_i * (1 - p_i) 가 됨. 
        - weight, p_i * (1 - p_i) 쓰면 거의 1을 못넘겨서 그냥 바로 서버림. 몇 번 나누지도 못하고 끝나서 underfitting 위험
        - classification에서 xgboost logloss 쓸 때 이걸 꺼주는 조치가 필요. 
    - regression 일 때는 cover = h_i = 1
- `gamma`: 크면 => tree 짧아짐. 
    - pruning 관련. 
    - default = 0
- `reg_lambda`: L2 reg hyperparam (Ridge)
    - 1개 tree가 너무 과한 영향 주지 못하게. 
- `reg_alpha`: L1 reg hyperparam (Lasso)
    - 불필요 feature, 강제로 0 만들어버려 feature selection 기능이 생김. 
    - 그걸 노리고 넣어둔 것. 
    - feature가 매우 많을 때 적용 권장. 
- 조언: `gamma`, `reg_lambda`, `reg_alpha`, 튜닝 효과 그리 크지 않고 너무 조합이 많으므로 그냥 디폴트 써도 무방. 
- **`max_depth`**: 가장 중요한 파라미터. 깊이를 설정하는 것. 
    - default = 6
    - 이것부터 튜닝하는 것이 좋다. 
- `subsample`: row sampling 비율. 
    - 대략 0.6~0.9 정도 사용. 
- `colsample_bytree`:
    - default = 1
    - col sampling이 row sampling 보다 시간을 많이줄여줌. (차피 col 을 나누는 것이 tree가 하는거니까. )
- `tree_method`: 
    - approx, hist, gpu_hist 가 적당히 쓰면서 tree 만듦. 
    - excat 는 오래걸림. 
- `objective`: 
    - classification에서 logloss를 많이 쓴다. 
- `base_score`: 
    - default는 0.5 지만, Y의 평균을 쓰는 것이 나을 수 있음. 
- `eval_metric`: 
    - CV 때는 평가점수 나오고, 이걸 바탕으로 validation 최적화 수행. 
- `early_stopping_rounds`: elbow 를 찾아 어디서 끊을지. 
    - 개선이 없을 시 얼마나 더 참을 것인가? 
    - `n_est`의 10% 정도를 권장. 
    - validation 용 fit() 매서드에서 eval_set= 에 지정해줘야 함. 
    - 훈련 끝난 것은 `.best_ntree_limit` property가 들어가있음. 이걸 이용해 최적 갯수를 확인 할 수 있고, 이 최적 갯수가 predict() 에서도 그대로 사용됨. 

### XGBoost 모델 학습과 예측

### 주요 매소드

- `fit()`
- `predict()`
- `eval_result()`

### 주요 속성

- `feature_importance` 

## 코드 리뷰

- xgb 기본 baseline으로 gbtree, binary:logistic lr 0.1 깔고간다. 
- param set 만들어서 dict로 `RandomizedSearchCV` 에 넣어준다. 
    - scoring 여기선 accuracy 했지만 F1 하는게 좋다고 말씀. 
    - refit=True best hyperparam 조합을 찾고 전체 train에 넣어 최종 모델을 만들어 준다. 
- (내 질문 관련) early stopping을 배웠다. 
    - n_est 를 아주 크게 막 10000 쓰고 lr을 작게 0.01 해놓으면 ??? 
    - cv를 할 때 hyperparam tuning 과 겹치는데??? --> 
    - xgb 돌릴 때 2 스텝으로 나눠서 한다. 
        - n_estimator 넣어주고 제일 좋은거 찾아주면, 


        cv 단계에 early stopping 시키면 공정하지 않다. 어떤 hyper param 조합은 조금만 가고 멈추고, 어떤건 멀리 가고 멈추고. 
        똑같은 estimator 개수를 놔도 정말 성능이 같으냐! 라는 것을 제대로 비교 못함. n est 가 다 달라지니까. hyper param 조합마다 tree 갯수가 달라지니까. 
        그래서 먼저 hyper param 조합 찾을 때는 n est 를 fix 시키고 찾고 (이땐 n est 를 100, 500 같은거만 사용)
        그 다음 cv 에서 n est 를 10000 으로 올리고 lr 도 0.01 로 낮추고 해서 안정화 시키며 early stopping 시킨다고... 

        randomized cv, 여기선 validation data 안썼다. train 내에서 cv 돌림. 

        [15] 부터 최종 모델이 나온다. finalmodel. 
        xgb 에서 튜닝해둔 값들을 입력하고, n est 확 늘려서 10000 하고 lr 확 낮춰서 0.01 하고 
        여기서 early stoppage 를 적용

        그래서, 처음 돌릴 때는 xgbclassifier 에 eval metric 안넣고 random cv 따로 돌려준거고 (스텝1)
        두 번째 돌릴 때는 xgbclassifier 에 eval meric 넣고 n est 와 lr 만 두고 돌려줌. (스텝2)


## 질문

### XGBoost loss vs eval metric

- 질문: 
    - 데이터를 train validation test 로 split 했으면, validation으로 hyperparam tuning 하는거 아닌가? 
    - --> 설명듣고 이해함. 

### loss vs eval 

- 질문: 문제에 적절한 loss, eval 선택방법
    - loss는 학습이 되어야 하고 (미분가능, 점진적 아래로로) eval은 평평하거나 불연속해도 됨. 
    - 보통 eval은 정해져있는 편. loss는 내가 정해야. 
        - loss 종류에 대해 따로 공부해야. stats learning 그 책 추천해 주셨던거에 좀 있다. 이건 군데군데 있다. 
        - loss 부분 정리한건 gpt 한테 물어봐도 된다.
        - 어떤 데이터에 맞고 어떤 문제가 있고. 이런 부분들이 매우 자세함. 

### 왜 결과가 이렇게 다를까요 저희가 한거랑? 


------
시험 준비는 여기까지. 

## SHAP

- feature importance보다 해석 면에서 나은 SHAP 개념 알고가자. 내가 부탁해서 다뤄주신다. 감사... 
- 우리가 구한 feature importance 는 설명하는데 좀 한계가 있다. 
    - 어떤 output이 나왔을 때, 여기에 대해 얼마나 기여를 했는가? 를 알 수 없다. --> 이게 무슨 말이지... 

- 원래 게임이론에서 왓다고 함. 
