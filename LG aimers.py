import os
from pprint import pprint
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap
import optuna
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
ROOT_DIR = "/Users/kuejealee/Study/LG Aimers/hackathon"
file_path_train = os.path.join(ROOT_DIR, "train.csv")  
train_data = pd.read_csv(file_path_train, sep=',', encoding='utf-8')
file_path_test = os.path.join(ROOT_DIR, 'test.csv')
test_data = pd.read_csv(file_path_test, sep=',', encoding='utf-8')

# pprint(train_data)

# 데이터 전처리 함수
def preprocessor(data):

    # (1) 결측치 제거
    drop_list = ['Receip No Collect Result_Fill2',
                 'Receip No Collect Result_Dam',
                 'Receip No Collect Result_Fill1',
                 'Production Qty Collect Result_Dam',
                 'Production Qty Collect Result_Fill1',
                 'Production Qty Collect Result_Fill2',
                 'PalletID Collect Result_Dam',
                 'PalletID Collect Result_Fill1',
                 'PalletID Collect Result_Fill2',
                 ]

    for c in data.columns:
        unique_val = data[c].dropna().unique() #결측치를 제거(.dropna) 했을 때 고윳값 개수(.unique)

        # 1) 모든 값이 null인 경우
        if len(unique_val) == 0: 
            drop_list.append(c)
        # 2) 모든 값이 동일한 경우
        elif len(unique_val) == 1 and data[c].isnull().sum() != len(data): 
            drop_list.append(c)


    # (2) 결측치 대체
    # 결측치 처리 함수 정의: KNN
    def missing_value_replacement(missing_value_column, a, b):
        # 숫자가 아닌 열 NaN값 처리
        missing_value_column = missing_value_column.apply(pd.to_numeric, errors='coerce')

        # 데이터프레임 생성
        KNN_data = {
            'target': missing_value_column,
            'feature1': a,
            'feature2': b
        }

        KNN_df = pd.DataFrame(KNN_data)

        # 결측치 제거 후 train data 생성
        KNN_train_data = KNN_df.dropna()

        # 입력값과 출력값 분할
        X = KNN_train_data[['feature1', 'feature2']]
        y = KNN_train_data['target']    

        # train, text 데이터로 분할
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

        # 학습
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(x_train,y_train)

        # 성능 테스트
        y_test_pred = knn.predict(x_test)

        # 성능 평가
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)

        # print(f"(MSE): {mse}")
        # print(f"(MAE): {mae}")

        # 실제 NaN값 예측 후 저장
        KNN_pred_data = KNN_df[KNN_df['target'].isnull()]

        y_pred = knn.predict(KNN_pred_data[['feature1','feature2']])

        # 원래의 DataFrame에 결측치 있는 부분만 대체
        result_column = missing_value_column.copy()
        result_column[KNN_pred_data.index] = y_pred

        return result_column

    # 결측치가 10% 이상인 열 knn 예측값으로 변경
    data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam']= missing_value_replacement(
        data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'],
        data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Dam'],
        data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Dam'],
        )
    data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1']= missing_value_replacement(
        data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1'],
        data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1'],
        data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1'],
        )
    data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2']= missing_value_replacement(
        data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'],
        data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill2'],
        data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill2'],
        )
            
    
    # (3) 데이터 정제
    # Workorder 숫자 데이터로 변환
    label_encoder = LabelEncoder()
    data['Workorder_Dam'] = label_encoder.fit_transform(data['Workorder_Dam'])

    # target값 숫자로 이진분류:
    data['target'] = data['target'].map({'Normal': 1, 'AbNormal': 0})

    # 데이터프레임의 각 열이 숫자인지 확인
    def is_numeric_series(series):
        return pd.api.types.is_numeric_dtype(series)
    
    # 문자열이 포함된 열 찾기
    non_numeric_columns = [col for col in data.columns if not is_numeric_series(data[col])]
    drop_list=drop_list+non_numeric_columns
    
     # 삭제할 데이터 삭제
    data = data.drop(columns=drop_list)
    print('데이터 삭제 완료')
    
    # # 데이터 분석을 위한 파일 저장
    # output_filename = os.path.join(ROOT_DIR, f"processed_data.csv")
    # data.to_csv(output_filename, index=False)
    # print(f"처리된 데이터프레임이 파일로 저장되었습니다.")


    # (4) 새로운 feature 만들기
    # 1) pca로 새로운 feature 생성
    # pca 생성 함수 정의
    def pca_creator(feature_name, *args):
        pca_data = pd.concat(args, axis=1)
        scaler = StandardScaler()
        pca_data = scaler.fit_transform(pca_data)
        pca = PCA(n_components=1)
        pca_data = pca.fit_transform(pca_data)

        data[f'new_feature_{feature_name}'] = pca_data

    # pca 진행
    print('PCA 시작')
    pca_creator('HNC X ACR_Dam', 
                data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Dam']
                )
    pca_creator('HNC Y ACR_Dam', 
                data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Dam']
                )
    pca_creator('HNC Z ACR_Dam', 
                data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Dam']
                )
    pca_creator('HNC X ACR_Fill1', 
                data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1'],
                data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1'],
                data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1']
                )
    pca_creator('HNC Y ACR_Fill1', 
                data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill1'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill1']
                )
    pca_creator('HNC Z ACR_Fill1', 
                data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1']
                )
    pca_creator('HNC X ACR_Fill2', 
                data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'],
                data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill2'],
                data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2']
                )
    pca_creator('HNC Y ACR_Fill2', 
                data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill2'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill2'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill2']
                )
    pca_creator('HNC Z ACR_Fill2', 
                data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill2'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill2'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill2']
                )
    pca_creator('(Stage2)_Dam',
                data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Dam'],
                data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Dam'],
                data['Dispense Volume(Stage2) Collect Result_Dam']    
                )
    
    # 2) 새로운 feature 생성(pca X)
    data['new_feature_Dam_Stage2_add'] = data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam']+data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Dam']+data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Dam']+data['Dispense Volume(Stage2) Collect Result_Dam']+data['Dispense Volume(Stage2) Collect Result_Dam']
    data['new_feature_Dam_discharged_distance'] = (data['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam']+data['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam']+data['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam'])*data['DISCHARGED SPEED OF RESIN Collect Result_Dam']
    data['new_feature_Dam_Pressure Unit Time difference'] = data['Chamber Temp. Unit Time_AutoClave']-(data['1st Pressure 1st Pressure Unit Time_AutoClave']+data['2nd Pressure Unit Time_AutoClave']+data['3rd Pressure Unit Time_AutoClave'])

    return(data)

# 데이터 전처리
print('------------1. 데이터 전처리------------')
print("1) train 데이터 전처리 시작")
train_data = preprocessor(train_data)

print('완료\n')

print("2) test 데이터 전처리 시작")
test_data = preprocessor(test_data)
print('완료\n')

# 언더샘플링
print('3) 언더샘플링 시작')
normal_ratio = 1.5 

df_normal = train_data[train_data["target"] == 1]
df_abnormal = train_data[train_data["target"] == 0]

num_normal = len(df_normal)
num_abnormal = len(df_abnormal)

df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=42)
train_data = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
train_data.value_counts("target")
print('완료\n')

# feature와 target 분리
print('4) feature와 target 값 분리')
X = train_data.drop('target', axis=1)  
y = train_data['target']      
print(f"전처리 후 데이터 크기: {X.shape}, {y.shape}")
print('완료\n')

# 훈련 데이터와 테스트 데이터 분리
print('5) train과 test 데이터로 분리')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('완료\n')

print('------------2. optuna 정의------------')
def objective(trial):
    param={
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.5, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
    }
    XGBoost_model = XGBClassifier(**param)
    XGBoost_model.fit(X_train,y_train, eval_set = [(X_test,y_test)], verbose=False)

    preds = XGBoost_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"before Accuracy: {accuracy}")

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print('------------optuna 최적화 끝------------')

bestparams = study.best_params

# 최적의 하이퍼파라미터 출력
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}= {},".format(key, value))


# 모델 정의
model = XGBClassifier(**bestparams)

# 모델 학습
print('\n------------3. 모델 학습------------')
   
print('1) 모델 학습')
model.fit(X_train,y_train)
print('완료\n')

print('2) 예측')
# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test) 
print('완료\n')

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("------------4. 모델 평가------------")
print(f'Accuracy: {accuracy* 100:.2f}%')
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}") 
print("\n")

# SHAP 값 계산
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train,)

# 특성 중요도 추출
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': shap_values.mean(axis=0).values})

pd.set_option('display.max_colwidth', None)

# 중요도 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df[:30], '\n')
