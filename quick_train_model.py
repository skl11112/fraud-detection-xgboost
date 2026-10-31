"""
快速生成模型脚本 - 跳过可视化，直接训练并保存模型
运行方式: python quick_train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb

PROJECT_DIR = r'C:\Users\季天源\Desktop\智算'
DATA_FILE = os.path.join(PROJECT_DIR, 'fraud_data.csv', 'fraud_data.csv')

print("=" * 60)
print("快速模型训练脚本")
print("=" * 60)

# 1. 加载数据
print("\n[1/5] 加载数据...")
df = pd.read_csv(DATA_FILE)
print(f"数据形状: {df.shape}")

# 2. 准备特征和标签
print("\n[2/5] 准备特征...")
y = df['label']
X = df.drop(columns=['label'])
feature_names = X.columns.tolist()
print(f"特征数: {len(feature_names)}")
print(f"样本数: {len(X)}")
print(f"标签分布: {y.value_counts().to_dict()}")

# 3. 构建 Pipeline
print("\n[3/5] 构建 Pipeline...")
scale_pos_weight = len(y[y == 0]) / max(len(y[y == 1]), 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# 4. 交叉验证评估
print("\n[4/5] 执行 5 折交叉验证...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, X, y,
    cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'roc_auc', 'f1'],
    return_estimator=False
)

print("\n交叉验证结果:")
print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
print(f"  AUC-ROC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"  F1:        {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

# 5. 全量训练并保存
print("\n[5/5] 全量训练并保存模型...")
pipeline.fit(X, y)

model_path = os.path.join(PROJECT_DIR, 'model_pipeline.pkl')
joblib.dump(pipeline, model_path)
print(f"[OK] 模型已保存: {model_path}")

# 保存特征名称
feature_path = os.path.join(PROJECT_DIR, 'feature_names.txt')
with open(feature_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_names))
print(f"[OK] 特征名称已保存: {feature_path}")

print("\n" + "=" * 60)
print("训练完成！现在可以启动 API 服务了")
print("=" * 60)
print("\n启动命令:")
print("  uvicorn main:app --reload --host 127.0.0.1 --port 8000")
print("\n访问文档:")
print("  http://127.0.0.1:8000/docs")
print("\n⚠️ 免责声明：模型输出仅为风险预警，不能直接定性财务舞弊。")
