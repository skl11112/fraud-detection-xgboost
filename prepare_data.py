"""
数据准备脚本：从原始parquet数据构建欺诈检测数据集
输出: fraud_data.csv (约100条样本，平衡的正负样本)
"""
import pandas as pd
import numpy as np
import os

base_path = "C:/Users/季天源/Desktop/智算/fraud_data.csv/data"
output_path = "C:/Users/季天源/Desktop/智算/fraud_data.csv"

print("加载面板数据...")
panel = pd.read_parquet(os.path.join(base_path, "panel/panel.parquet"))

print(f"原始数据形状: {panel.shape}")

# 获取列名（处理编码问题）
cols = panel.columns.tolist()

# 关键列索引（根据之前的探索）
# 62: 违规标签 (target)
# 26: code (股票代码)
# 27: year (年份)
# 财务指标列: 1-25, 28-50

target_col = cols[62]  # 违规标签
code_col = cols[26]    # code
year_col = cols[27]    # year

print(f"目标列: {target_col}")
print(f"目标分布:\n{panel[target_col].value_counts(dropna=False)}")

# 选择用于建模的特征（财务报表指标）
# 排除字符串列、日期列、行业分类列
feature_cols = []
for i, col in enumerate(cols):
    if i in [26, 27, 44, 52, 53, 54]:  # 排除 code, year, 报告日期, sw_l1, sw_l2, 审计意见
        continue
    if panel[col].dtype in ['float64', 'int64', 'Int64']:
        feature_cols.append(col)

print(f"\n选择特征数: {len(feature_cols)}")

# 过滤有效样本（目标变量非空）
df_valid = panel[panel[target_col].notna()].copy()
print(f"有效样本数: {len(df_valid)}")

# 分离正负样本
df_fraud = df_valid[df_valid[target_col] == 1]
df_normal = df_valid[df_valid[target_col] == 0]

print(f"欺诈样本: {len(df_fraud)}")
print(f"正常样本: {len(df_normal)}")

# 采样构建平衡数据集（约100条）
# 欺诈样本: 40条，正常样本: 60条
np.random.seed(42)
n_fraud = min(40, len(df_fraud))
n_normal = min(60, len(df_normal))

df_fraud_sample = df_fraud.sample(n=n_fraud, random_state=42)
df_normal_sample = df_normal.sample(n=n_normal, random_state=42)

# 合并并打乱
df_final = pd.concat([df_fraud_sample, df_normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n最终数据集形状: {df_final.shape}")
print(f"最终目标分布:\n{df_final[target_col].value_counts()}")

# 选择关键特征（保留业务含义明确的财务指标）
# 资产负债表、利润表、现金流量表核心指标
key_features = [
    cols[1],   # 货币资金
    cols[2],   # 应收账款净额
    cols[3],   # 存货
    cols[6],   # 流动资产合计
    cols[7],   # 固定资产净额
    cols[10],  # 无形资产
    cols[11],  # 资产总计
    cols[17],  # 流动负债合计
    cols[21],  # 负债合计
    cols[22],  # 实收资本(股本)
    cols[23],  # 盈余公积
    cols[24],  # 未分配利润
    cols[25],  # 股东权益合计
    cols[28],  # 营业总收入
    cols[30],  # 营业成本
    cols[31],  # 销售费用
    cols[32],  # 管理费用
    cols[33],  # 财务费用
    cols[34],  # 投资收益
    cols[36],  # 资产减值损失
    cols[38],  # 营业利润
    cols[39],  # 利润总额
    cols[40],  # 所得税费用
    cols[41],  # 净利润
    cols[42],  # 归属于母公司净利润
    cols[45],  # 经营活动现金净流量
    cols[46],  # 支付的所得税
    cols[47],  # 销售商品收到的现金
    cols[49],  # 投资活动现金净流量
    cols[50],  # 筹资活动现金净流量
    cols[51],  # 审计缺失标志
    cols[57],  # 是否ST
    cols[58],  # 违规状态
    cols[63],  # 是否金融业
    cols[64],  # 审计师通过
]

# 过滤存在的特征
key_features = [f for f in key_features if f in df_final.columns]

# 构建最终数据集
df_output = df_final[key_features + [target_col]].copy()

# 重命名目标列为 'label'
df_output = df_output.rename(columns={target_col: 'label'})

# 保存
output_file = os.path.join(output_path, "fraud_data.csv") if os.path.isdir(output_path) else "C:/Users/季天源/Desktop/智算/fraud_data.csv"
# 实际上 fraud_data.csv 是一个目录，所以保存在里面
output_file = "C:/Users/季天源/Desktop/智算/fraud_data.csv/fraud_data.csv"
df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n数据集已保存到: {output_file}")
print(f"特征数: {len(key_features)}")
print(f"样本数: {len(df_output)}")
print(f"\n特征列表:")
for i, col in enumerate(df_output.columns):
    if col != 'label':
        print(f"  {i+1}. {col}")
print(f"\n目标变量: label (0=正常, 1=欺诈)")
print(f"标签分布:\n{df_output['label'].value_counts()}")

# 保存特征名称供后续使用
feature_names = [c for c in df_output.columns if c != 'label']
with open("C:/Users/季天源/Desktop/智算/feature_names.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_names))

print(f"\n特征名称已保存到: feature_names.txt")
