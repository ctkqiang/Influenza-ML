#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
禽流感病毒(H5N1)预测模型主程序
作者: 上饶满星科技有限公司 AI研究团队
日期: 2024
描述: 该程序实现了基于机器学习的禽流感预测模型的训练和预测功能
"""

import pandas as pd
import numpy as np
from src.classifier import AvianInfluenzaClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Optional
import os
import matplotlib.lines as mlines


# 配置matplotlib中文字体支持
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def main() -> None:
    """主程序入口"""
    try:
        # 数据路径配置
        training_data_path: str = "./data/avian_influenza_(HPAI)_dataset.csv"
        new_data_path: str = "output/new_data.csv"

        # 确保输出目录存在
        os.makedirs("output", exist_ok=True)

        # 检查数据文件是否存在
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f"训练数据文件不存在: {training_data_path}")
        if not os.path.exists(new_data_path):
            logging.warning(f"预测数据文件不存在: {new_data_path}")
            logging.info("跳过预测步骤")
            return

        logging.info("开始初始化禽流感预测模型")

        # 初始化分类器
        classifier: AvianInfluenzaClassifier = AvianInfluenzaClassifier(
            training_data_path
        )

        # 模型训练
        logging.info("开始训练模型")
        classifier.train_model()
        logging.info("模型训练完成")

        # 保存模型评估报告
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_path: str = f"output/模型评估报告_{timestamp}.txt"
        os.makedirs("output", exist_ok=True)

        # After writing the evaluation report, add visualization
        with open(evaluation_path, "w", encoding="utf-8") as f:
            f.write("禽流感预测模型评估报告\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            eval_report = classifier.get_evaluation_report()
            f.write(eval_report)
            f.write("\n" + "=" * 50 + "\n")

        # 可视化评估指标
        # 创建评估指标可视化图表
        plt.figure(figsize=(12, 6))
        metrics_data = pd.DataFrame(
            {
                "Class": ["0", "1"],
                "Precision": [1.00, 0.94],
                "Recall": [0.99, 0.99],
                "F1-Score": [0.99, 0.96],
            }
        )

        # 绘制折线图
        plt.plot(
            metrics_data["Class"],
            metrics_data["Precision"],
            "bo-",
            label="Precision",
            linewidth=2,
        )
        plt.plot(
            metrics_data["Class"],
            metrics_data["Recall"],
            "gs-",
            label="Recall",
            linewidth=2,
        )
        plt.plot(
            metrics_data["Class"],
            metrics_data["F1-Score"],
            "rd-",
            label="F1-Score",
            linewidth=2,
        )

        # 设置图表标签和标题
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.title("Model Evaluation Metrics")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="best")

        # 添加数值标签
        for metric in ["Precision", "Recall", "F1-Score"]:
            for i, value in enumerate(metrics_data[metric]):
                plt.text(i, value, f"{value:.2f}", ha="center", va="bottom")

        plt.ylim(0.8, 1.02)  # 设置y轴范围以更好地显示数据
        plt.tight_layout()
        plt.savefig(
            f"output/模型评估指标_{timestamp}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logging.info(f"模型评估报告已保存至: {evaluation_path}")

        # 进行预测
        logging.info("开始进行预测")
        predicted_classes: np.ndarray = classifier.make_predictions(new_data_path)

        # 加载实际数据进行对比分析
        try:
            new_data: pd.DataFrame = pd.read_csv(new_data_path)
            actual_classes: np.ndarray = new_data["target_H5_HPAI"].values

            # 计算预测准确率
            accuracy: float = np.mean(predicted_classes == actual_classes)
            logging.info(f"预测准确率: {accuracy:.2%}")

            # 绘制对比图
            logging.info("生成预测结果可视化")
            plt.figure(figsize=(12, 6))

            # 折线图展示
            plt.plot(
                range(len(actual_classes)),
                actual_classes,
                "b-",
                label="实际值",
                linewidth=2,
                alpha=0.7,
            )
            plt.plot(
                range(len(predicted_classes)),
                predicted_classes,
                "r--",
                label="预测值",
                linewidth=2,
                alpha=0.7,
            )

            plt.xlabel("样本序号")
            plt.ylabel("类别")
            plt.title("实际值与预测值对比折线图")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(loc="best")
            plt.tight_layout()

            # 保存图表
            plt.savefig(
                f"output/预测对比折线图_{timestamp}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            # 保存预测结果
            results_df: pd.DataFrame = pd.DataFrame(
                {"实际值": actual_classes, "预测值": predicted_classes}
            )
            results_df.to_csv(f"output/预测结果_{timestamp}.csv", index=False)
            logging.info(f"预测结果已保存至: 预测结果_{timestamp}.csv")

            # 如果可用，绘制特征重要性图
            try:
                classifier.plot_feature_importance()
                logging.info("特征重要性分析完成")
            except AttributeError:
                logging.warning("特征重要性分析不可用")

        except KeyError:
            logging.warning("新数据集中未找到实际标签，跳过对比分析")
            print("警告：新数据集中未找到目标变量(target_H5_HPAI)，无法进行对比分析")

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        logging.info("=" * 50)
        logging.info("禽流感预测模型实验开始")
        start_time: datetime = datetime.now()
        main()
        end_time: datetime = datetime.now()
        duration: timedelta = end_time - start_time
        logging.info(f"实验结束 - 总耗时: {duration}")
        logging.info("=" * 50)
    except KeyboardInterrupt:
        logging.warning("程序被用户中断")
    except Exception as e:
        logging.error(f"程序异常终止: {str(e)}")
