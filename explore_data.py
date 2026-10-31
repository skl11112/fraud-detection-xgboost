"""
数据探索脚本：了解所有数据文件的结构
"""
import pandas as pd
import os

base_path = "C:/Users/季天源/Desktop/智算/fraud_data.csv/data"

print("=" * 80)
print("PARQUET 文件探索")
print("=" * 80)

parquet_files = [
    "parquet/audit.parquet",
    "parquet/balance.parquet",
    "parquet/cashflow.parquet",
    "parquet/income.parquet",
    "parquet/industry_sw.parquet",
    "parquet/st_change.parquet",
    "parquet/violation.parquet",
    "panel/panel.parquet"
]

for file in parquet_files:
    full_path = os.path.join(base_path, file)
    if os.path.exists(full_path):
        try:
            df = pd.read_parquet(full_path)
            print(f"\n{file}:")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns[:15])}{'...' if len(df.columns) > 15 else ''}")
            print(f"  前3行:")
            print(df.head(3).to_string(index=False))
        except Exception as e:
            print(f"\n{file}: 读取错误 - {e}")

print("\n" + "=" * 80)
print("XLSX 文件探索")
print("=" * 80)

xlsx_files = [
    "caiwubaobiao/lirunbiao/FS_Comins.xlsx",
    "caiwubaobiao/xianjinliuliangbiao/FS_Comscfd.xlsx",
    "caiwubaobiao/zichanfuzhaibiao/FS_Combas.xlsx",
    "gongsifenlei/STK_IndustryClassAnl.xlsx",
    "shenjiyijian/FIN_Audit.xlsx",
    "teshuchuli/SPT_Trdchg.xlsx",
    "weiguichufa/STK_Violation_Main.xlsx"
]

for file in xlsx_files:
    full_path = os.path.join(base_path, file)
    if os.path.exists(full_path):
        try:
            df = pd.read_excel(full_path)
            print(f"\n{file}:")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns[:15])}{'...' if len(df.columns) > 15 else ''}")
            print(f"  前3行:")
            print(df.head(3).to_string(index=False))
        except Exception as e:
            print(f"\n{file}: 读取错误 - {e}")

print("\n" + "=" * 80)
print("数据关联分析")
print("=" * 80)

# 读取关键表分析关联
try:
    violation = pd.read_parquet(os.path.join(base_path, "parquet/violation.parquet"))
    balance = pd.read_parquet(os.path.join(base_path, "parquet/balance.parquet"))
    income = pd.read_parquet(os.path.join(base_path, "parquet/income.parquet"))
    
    print("\nviolation 表关键字段:")
    print(violation.columns.tolist())
    print(f"\nviolation 样本数: {len(violation)}")
    if 'Stkcode' in violation.columns:
        print(f"唯一股票代码数: {violation['Stkcode'].nunique()}")
    
    print("\nbalance 表关键字段:")
    print(balance.columns.tolist()[:20])
    print(f"\nbalance 样本数: {len(balance)}")
    if 'Stkcode' in balance.columns:
        print(f"唯一股票代码数: {balance['Stkcode'].nunique()}")
        
    print("\nincome 表关键字段:")
    print(income.columns.tolist()[:20])
    print(f"\nincome 样本数: {len(income)}")
    if 'Stkcode' in income.columns:
        print(f"唯一股票代码数: {income['Stkcode'].nunique()}")
        
except Exception as e:
    print(f"关联分析错误: {e}")
