# data.py
import streamlit as st
from datetime import datetime
from database import save_to_db, get_db_count # DB操作関数をインポート

# サンプルデータのリスト
SAMPLE_QUESTIONS_DATA = [
    {
        "question": "アカデミック・リンク・センターとはなんですか？",
        "answer": "アカデミック・リンク・センターは、学術的な支援を提供する施設で、学生が学業を成功させるためのリソースやサービスを提供します。",
        "correct_answer": "アカデミック・リンク・センターは、千葉大学にある図書館とともにある、学習支援のための教員組織で、自ら学ぶ学習者のためのリソースやサービス、その学習者を育てる教育学修支援専門職の養成をミッションにしています。",
        "feedback": "部分的に正確: 基本的な説明は正しいですが、おそらく、類推しているだけで、センターのことを話していません。",
        "is_correct": 0.5,
        "response_time": 1.2
    }
]

def create_sample_evaluation_data():
    """定義されたサンプルデータをデータベースに保存する"""
    try:
        count_before = get_db_count()
        added_count = 0
        # 各サンプルをデータベースに保存
        for item in SAMPLE_QUESTIONS_DATA:
            # save_to_dbが必要な引数のみ渡す
            save_to_db(
                question=item["question"],
                answer=item["answer"],
                feedback=item["feedback"],
                correct_answer=item["correct_answer"],
                is_correct=item["is_correct"],
                response_time=item["response_time"]
            )
            added_count += 1

        count_after = get_db_count()
        st.success(f"{added_count} 件のサンプル評価データが正常に追加されました。(合計: {count_after} 件)")

    except Exception as e:
        st.error(f"サンプルデータの作成中にエラーが発生しました: {e}")
        print(f"エラー詳細: {e}") # コンソールにも出力

def ensure_initial_data():
    """データベースが空の場合に初期サンプルデータを投入する"""
    if get_db_count() == 0:
        st.info("データベースが空です。初期サンプルデータを投入します。")
        create_sample_evaluation_data()

from database import get_all_evaluations

def get_all_correct_answers():
    """
    データベースおよびサンプルデータからcorrect_answerを取得
    """
    # サンプルデータから取得
    answers = [item["correct_answer"] for item in SAMPLE_QUESTIONS_DATA]
    # データベースから取得
    try:
        records = get_all_evaluations()
        answers += [r["correct_answer"] for r in records]
    except Exception:
        pass
    return answers