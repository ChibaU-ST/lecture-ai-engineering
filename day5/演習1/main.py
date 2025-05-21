import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature

# データ準備
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = "data/Titanic.csv"
    data = pd.read_csv(path)

    # 欠損値の補完 (inplace=Trueを使わない方法)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# -------- 特徴量エンジニアリング（追加） --------
    # 特徴量エンジニアリング
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        # 名前から敬称（Title）を抽出
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # 敬称をグループ化
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = LabelEncoder().fit_transform(data['Title'])
    
    # キャビン情報からデッキを抽出
    data['Deck'] = data['Cabin'].str[0].fillna('U')
    deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
    data['Deck'] = data['Deck'].map(deck_mapping)
    
    # 運賃のビン化（等分位数）
    data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)
    
    # 年齢のビン化
    data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 18, 65, 100], labels=[0, 1, 2, 3])
    
    # 既存の特徴量エンジニアリング
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
# -------- 特徴量エンジニアリング（ここまで） --------
    
    # 家族サイズから単独旅行者かを判定
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    # カテゴリ変数の数値化
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    data["Embarked"] = LabelEncoder().fit_transform(data["Embarked"])

    # 浮動小数点型に変換
    for col in ["Pclass", "Sex", "Age", "Fare", "FamilySize", "Embarked", "Survived"]:
        data[col] = data[col].astype(float)

    # 説明変数と目的変数
    X = data[["Pclass", "Sex", "Age", "Fare", "FamilySize", "Embarked"]]
    y = data["Survived"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# モデルの現在の状態を確認
def analyze_current_model():
    path = "data/Titanic.csv"
    data = pd.read_csv(path)
    
    print("データサンプル:")
    print(data.head())
    
    print("\nデータの基本統計:")
    print(data.describe())
    
    print("\n欠損値の状況:")
    print(data.isnull().sum())
    
    # 特徴量の重要度確認
    X_train, X_test, y_train, y_test = prepare_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徴量の重要度:")
    print(importances)
    
    # 現状のモデル性能
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n交差検証スコア: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return data


# 学習と評価
def train_and_evaluate(
    X_train, X_test, y_train, y_test, random_state=42
):
    # ハイパーパラメータ候補
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    base_clf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(
        base_clf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return best_model, accuracy, grid.best_params_


# モデル保存
def log_model(model, accuracy, params):
    with mlflow.start_run():
        # パラメータをログ
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # メトリクスをログ
        mlflow.log_metric("accuracy", accuracy)

        # モデルのシグネチャを推論
        signature = infer_signature(X_train, model.predict(X_train))

        # モデルを保存
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:5],  # 入力例を指定
        )
        # accurecyとparmsは改行して表示
        print(f"モデルのログ記録値 \naccuracy: {accuracy}\nparams: {params}")


# メイン処理
if __name__ == "__main__":
    # ランダム要素の設定
    test_size = round(
        random.uniform(0.1, 0.3), 2
    )  # 10%〜30%の範囲でテストサイズをランダム化
    data_random_state = random.randint(1, 100)
    model_random_state = random.randint(1, 100)
    n_estimators = random.randint(50, 200)
    max_depth = random.choice([None, 3, 5, 10, 15])

    # パラメータ辞書の作成
    params = {
        "test_size": test_size,
        "data_random_state": data_random_state,
        "model_random_state": model_random_state,
        "n_estimators": n_estimators,
        "max_depth": "None" if max_depth is None else max_depth,
    }

    # データ準備
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=test_size, random_state=data_random_state
    )

    # 学習と評価
    model, accuracy, best_params = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=model_random_state,
    )

    # モデル保存
    # pass best_params to log_model
    log_model(model, accuracy, {**params, **best_params})

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"titanic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"モデルを {model_path} に保存しました")
