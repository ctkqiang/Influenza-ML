import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Optional, List
from datetime import datetime
import os


class AvianInfluenzaClassifier:
    """
    禽流感分类器类
    实现了数据预处理、模型训练、预测和可视化功能
    采用决策树算法和SMOTE过采样技术处理不平衡数据
    """

    def __init__(self, data_path: str) -> None:
        super(AvianInfluenzaClassifier, self).__init__()
        self.data_path = data_path
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None
        self.evaluation_report: Optional[str] = None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols: pd.Index = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        return df

    def train_model(self) -> None:
        df: pd.DataFrame = pd.read_csv(self.data_path)
        X: pd.DataFrame = df.drop(["_id", "target_H5_HPAI"], axis=1)
        y: pd.Series = df["target_H5_HPAI"]

        X = self.preprocess_data(X)

        X_train: np.ndarray
        X_test: np.ndarray
        y_train: np.ndarray
        y_test: np.ndarray
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        smote: SMOTE = SMOTE(random_state=42)
        X_train_resampled: np.ndarray
        y_train_resampled: np.ndarray
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # 训练决策树模型
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train_resampled, y_train_resampled)

        # 计算特征重要性
        self.feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # 评估模型性能
        y_pred = self.model.predict(X_test)
        self.evaluation_report = classification_report(y_test, y_pred)
        print("模型评估报告:")
        print(self.evaluation_report)

    def get_evaluation_report(self) -> str:
        """
        获取模型评估报告
        返回:
            str: 包含精确度、召回率和F1分数的评估报告
        """
        if not hasattr(self, "evaluation_report"):
            raise AttributeError("模型尚未训练，无法获取评估报告")
        return self.evaluation_report

    def make_predictions(self, new_data_path: str) -> np.ndarray:
        new_df: pd.DataFrame = pd.read_csv(new_data_path)
        new_X: pd.DataFrame = (
            new_df.drop(["_id"], axis=1) if "_id" in new_df.columns else new_df
        )
        new_X = self.preprocess_data(new_X)

        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train_model() 方法")

        predictions: np.ndarray = self.model.predict(new_X)
        return predictions

        def plot_comparison(self, actual: np.ndarray, predicted: np.ndarray) -> None:
            try:
                # 创建图表保存目录
                plots_dir: str = "output/plots"
                os.makedirs(plots_dir, exist_ok=True)
                timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 混淆矩阵图
                cm: np.ndarray = confusion_matrix(actual, predicted)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("混淆矩阵")
                plt.ylabel("实际类别")
                plt.xlabel("预测类别")
                print(
                    f"Saving confusion matrix plot to {plots_dir}/混淆矩阵_{timestamp}.png"
                )
                plt.savefig(
                    f"{plots_dir}/混淆矩阵_{timestamp}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # 类别分布对比图
                unique_classes = np.unique(np.concatenate([actual, predicted]))
                actual_counts = [np.sum(actual == c) for c in unique_classes]
                predicted_counts = [np.sum(predicted == c) for c in unique_classes]

                x = np.arange(len(unique_classes))
                width = 0.35

                fig, ax = plt.subplots(figsize=(10, 6))
                rects1 = ax.bar(x - width / 2, actual_counts, width, label="实际值")
                rects2 = ax.bar(x + width / 2, predicted_counts, width, label="预测值")

                ax.set_ylabel("样本数量")
                ax.set_title("实际值与预测值的类别分布对比")
                ax.set_xticks(x)
                ax.set_xticklabels(unique_classes)
                ax.legend()

                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(
                            f"{height}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                        )

                autolabel(rects1)
                autolabel(rects2)
                print(
                    f"Saving class distribution plot to {plots_dir}/类别分布对比_{timestamp}.png"
                )
                plt.tight_layout()
                plt.savefig(
                    f"{plots_dir}/类别分布对比_{timestamp}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as e:
                print(f"An error occurred during plotting: {e}")

    def plot_feature_importance(self) -> None:
        if self.feature_importance is None:
            raise ValueError("请先训练模型以获取特征重要性")

        plots_dir: str = "output/plots"
        os.makedirs(plots_dir, exist_ok=True)
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure(figsize=(12, 6))
        sns.barplot(x="importance", y="feature", data=self.feature_importance.head(10))
        plt.title("前10个最重要特征")
        plt.xlabel("特征重要性")
        plt.ylabel("特征名称")
        plt.tight_layout()
        plt.savefig(
            f"{plots_dir}/特征重要性_{timestamp}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def main():
        data_path = "your_data_path.csv"
        classifier = AvianInfluenzaClassifier(data_path)
        classifier.train_model()
        actual = np.array([...])
        predicted = np.array([...])
        classifier.plot_comparison(actual, predicted)

        classifier.plot_feature_importance()
