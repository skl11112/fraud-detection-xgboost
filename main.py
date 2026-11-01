"""
财务舞弊风险检测 FastAPI 服务
启动命令: uvicorn main:app --reload --host 127.0.0.1 --port 8000
接口文档: http://127.0.0.1:8000/docs
"""
import os
import traceback
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import tempfile

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "model_pipeline.pkl")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names.txt")

app = FastAPI(
    title="财务舞弊风险检测 API",
    description="基于 XGBoost 的财务舞弊风险预警接口，提供风险概率预测及 SHAP 可解释性分析。",
    version="1.0.0"
)

pipeline = None
xgb_model = None
imputer = None
scaler = None
explainer = None
feature_names = None


def load_model():
    """加载模型和初始化 SHAP 解释器"""
    global pipeline, xgb_model, imputer, scaler, explainer, feature_names

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    xgb_model = pipeline.named_steps["classifier"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    explainer = shap.TreeExplainer(xgb_model)

    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            feature_names = [line.strip() for line in f if line.strip()]
    else:
        feature_names = [f"f_{i}" for i in range(xgb_model.n_features_in_)]

    print(f"模型加载完成: {MODEL_PATH}")
    print(f"特征数量: {len(feature_names)}")
    print(f"SHAP TreeExplainer 初始化完成")


@app.on_event("startup")
async def startup_event():
    load_model()


class 预测请求(BaseModel):
    """请求体模型：输入特征值"""
    features: Dict[str, float] = Field(
        ...,
        description="特征名到特征值的映射，键为特征名称，值为对应的数值"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    feature_names[0] if feature_names else "货币资金": 1.0e8,
                    feature_names[1] if feature_names else "应收账款净额": 2.0e8,
                }
            }
        }


class SHAP解释项(BaseModel):
    """单个特征的 SHAP 解释"""
    特征名称: str = Field(..., description="特征名称")
    特征值: float = Field(..., description="特征原始值")
    SHAP值: float = Field(..., description="SHAP 值（正值推动高风险，负值推动低风险）")
    SHAP绝对值: float = Field(..., description="SHAP 值绝对值")


class 预测响应(BaseModel):
    """预测响应"""
    风险概率: float = Field(..., description="风险概率 (0~1)")
    风险判定: str = Field(..., description="风险判定结果: '高风险' 或 '正常'")
    模型基准值: float = Field(..., description="模型基准值 (expected value)")
    SHAP解释: List[SHAP解释项] = Field(..., description="各特征 SHAP 解释列表")
    主要风险因素: List[str] = Field(default=[], description="Top 3 风险推动因素")
    免责声明: str = Field(
        default="⚠️ 本模型输出仅为风险预警参考，不能直接定性财务舞弊。",
        description="免责声明"
    )


class 错误响应(BaseModel):
    """错误响应"""
    错误: str
    详情: Optional[str] = None


@app.get("/", summary="服务状态")
async def root():
    """返回服务基本信息"""
    return {
        "服务名称": "财务舞弊风险检测 API",
        "版本": "1.0.0",
        "状态": "运行中",
        "文档地址": "/docs",
        "免责声明": "模型输出仅为风险预警，不能直接定性财务舞弊。"
    }


@app.get("/health", summary="健康检查")
async def health():
    """检查服务运行状态和模型加载情况"""
    return {
        "状态": "健康",
        "模型已加载": pipeline is not None,
        "特征数量": len(feature_names) if feature_names else 0
    }


@app.get("/features", summary="获取特征列表")
async def get_features():
    """返回模型使用的全部特征名称"""
    return {
        "特征数量": len(feature_names),
        "特征名称": feature_names
    }


@app.post("/predict_risk", response_model=预测响应, summary="风险预测",
          responses={400: {"model": 错误响应}})
async def predict_risk(request: 预测请求):
    """
    财务舞弊风险预测接口

    接收财务报表特征数据，返回:
    - 风险概率
    - 风险判定（正常/高风险）
    - SHAP 可解释性分析
    """
    try:
        input_features = request.features

        missing = [f for f in feature_names if f not in input_features]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"缺少特征: {missing[:5]}{'...' if len(missing) > 5 else ''} (共缺 {len(missing)} 个)"
            )

        values = [float(input_features[f]) for f in feature_names]
        X_input = pd.DataFrame([values], columns=feature_names)

        X_processed = scaler.transform(imputer.transform(X_input))

        risk_prob = float(pipeline.predict_proba(X_input)[0][1])
        threshold = 0.5
        risk_result = "高风险" if risk_prob >= threshold else "正常"

        sample_explainer = shap.TreeExplainer(xgb_model)
        shap_vals = sample_explainer.shap_values(X_processed)[0]
        base_val = float(sample_explainer.expected_value)

        shap_explain_list = []
        for j, fname in enumerate(feature_names):
            shap_explain_list.append(SHAP解释项(
                特征名称=fname,
                特征值=values[j],
                SHAP值=float(shap_vals[j]),
                SHAP绝对值=float(abs(shap_vals[j]))
            ))

        shap_explain_sorted = sorted(shap_explain_list, key=lambda x: x.SHAP绝对值, reverse=True)
        top_factors = [
            f"{item.特征名称} (SHAP={item.SHAP值:+.4f})"
            for item in shap_explain_sorted[:3]
        ]

        return 预测响应(
            风险概率=round(risk_prob, 6),
            风险判定=risk_result,
            模型基准值=round(base_val, 6),
            SHAP解释=shap_explain_list,
            主要风险因素=top_factors
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"预测失败: {str(e)}"
        )


@app.post("/batch_predict", summary="批量预测")
async def batch_predict(requests: List[预测请求]):
    """批量预测接口，支持同时预测多个样本"""
    results = []
    for req in requests:
        try:
            values = [float(req.features[f]) for f in feature_names]
            X_input = pd.DataFrame([values], columns=feature_names)
            risk_prob = float(pipeline.predict_proba(X_input)[0][1])
            results.append({
                "风险概率": round(risk_prob, 6),
                "风险判定": "高风险" if risk_prob >= 0.5 else "正常"
            })
        except Exception as e:
            results.append({"错误": str(e)})
    return {"结果": results}


# 财务报表列名映射（原始Excel列名 → 模型特征名）
COLUMN_MAPPING = {
    # 资产负债表
    "货币资金": "货币资金",
    "货币资金:": "货币资金",
    "应收账款净额": "应收账款净额",
    "应收账款": "应收账款净额",
    "应收票据及应收账款": "应收账款净额",
    "存货净额": "存货净额",
    "存货": "存货净额",
    "流动资产合计": "流动资产合计",
    "固定资产净额": "固定资产净额",
    "固定资产": "固定资产净额",
    "商誉净额": "商誉净额",
    "商誉": "商誉净额",
    "资产总计": "资产总计",
    "流动负债合计": "流动负债合计",
    "负债合计": "负债合计",
    "实收资本(或股本)": "实收资本(或股本)",
    "实收资本（或股本）": "实收资本(或股本)",
    "实收资本": "实收资本(或股本)",
    "股本": "实收资本(或股本)",
    "盈余公积": "盈余公积",
    "未分配利润": "未分配利润",
    "所有者权益合计": "所有者权益合计",
    "股东权益合计": "所有者权益合计",
    # 利润表
    "营业总收入": "营业总收入",
    "营业收入": "营业总收入",
    "营业成本": "营业成本",
    "销售费用": "销售费用",
    "管理费用": "管理费用",
    "财务费用": "财务费用",
    "投资收益": "投资收益",
    "资产减值损失": "资产减值损失",
    "营业利润": "营业利润",
    "利润总额": "利润总额",
    "减：所得税费用": "减：所得税费用",
    "所得税费用": "减：所得税费用",
    "净利润": "净利润",
    "归属于母公司所有者的净利润": "归属于母公司所有者的净利润",
    "归母净利润": "归属于母公司所有者的净利润",
    # 现金流量表
    "经营活动产生的现金流量净额": "经营活动产生的现金流量净额",
    "经营活动现金净流量": "经营活动产生的现金流量净额",
    "支付的各项税费": "支付的各项税费",
    "销售商品、提供劳务收到的现金": "销售商品、提供劳务收到的现金",
    "投资活动产生的现金流量净额": "投资活动产生的现金流量净额",
    "投资活动现金净流量": "投资活动产生的现金流量净额",
    "筹资活动产生的现金流量净额": "筹资活动产生的现金流量净额",
    "筹资活动现金净流量": "筹资活动产生的现金流量净额",
}


def 解析财务文件(file_path: str, file_name: str) -> Dict[str, float]:
    """解析上传的财务文件，提取特征值"""
    ext = os.path.splitext(file_name)[1].lower()

    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}，仅支持 .xlsx/.xls/.csv")

    if df.empty:
        raise HTTPException(status_code=400, detail="文件为空，无数据")

    # 取第一行数据（最新一期财报）
    row = df.iloc[0]
    extracted = {}

    for col in df.columns:
        col_str = str(col).strip()
        # 尝试精确匹配
        if col_str in COLUMN_MAPPING:
            model_feature = COLUMN_MAPPING[col_str]
            val = row[col]
            if pd.notna(val):
                extracted[model_feature] = float(val)
        else:
            # 尝试模糊匹配（去除空格、括号差异）
            for raw_name, model_name in COLUMN_MAPPING.items():
                if col_str.replace(' ', '').replace('（', '(').replace('）', ')') == \
                   raw_name.replace(' ', '').replace('（', '(').replace('）', ')'):
                    val = row[col]
                    if pd.notna(val):
                        extracted[model_name] = float(val)
                    break

    return extracted


class 文件预测响应(BaseModel):
    """文件上传预测响应"""
    文件名: str = Field(..., description="上传的文件名")
    提取特征数: int = Field(..., description="从文件中成功提取的特征数量")
    缺失特征数: int = Field(..., description="未能从文件中提取的特征数量")
    缺失特征: List[str] = Field(default=[], description="缺失的特征名称列表")
    风险概率: float = Field(..., description="风险概率 (0~1)")
    风险判定: str = Field(..., description="风险判定结果")
    模型基准值: float = Field(..., description="模型基准值")
    主要风险因素: List[str] = Field(default=[], description="Top 3 风险推动因素")
    免责声明: str = Field(
        default="⚠️ 本模型输出仅为风险预警参考，不能直接定性财务舞弊。",
        description="免责声明"
    )


@app.post("/upload_predict", summary="上传文件预测", response_model=文件预测响应)
async def upload_predict(file: UploadFile = File(..., description="上传财务报表文件（.xlsx/.xls/.csv）")):
    """
    上传财务报表文件进行风险预测

    支持上传资产负债表、利润表、现金流量表的 Excel 或 CSV 文件。
    系统会自动识别列名并提取特征，缺失的特征将由模型自动填充。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.xlsx', '.xls', '.csv']:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}，仅支持 .xlsx/.xls/.csv")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            extracted = 解析财务文件(tmp_path, file.filename)
        finally:
            os.unlink(tmp_path)

        matched_count = len(extracted)
        missing = [f for f in feature_names if f not in extracted]

        # 构建完整特征向量（缺失的用 NaN，由 Pipeline 的中位数填充处理）
        values = []
        for f in feature_names:
            if f in extracted:
                values.append(extracted[f])
            else:
                values.append(np.nan)

        X_input = pd.DataFrame([values], columns=feature_names)
        X_processed = scaler.transform(imputer.transform(X_input))

        risk_prob = float(pipeline.predict_proba(X_input)[0][1])
        risk_result = "高风险" if risk_prob >= 0.5 else "正常"

        sample_explainer = shap.TreeExplainer(xgb_model)
        shap_vals = sample_explainer.shap_values(X_processed)[0]
        base_val = float(sample_explainer.expected_value)

        shap_items = []
        for j, fname in enumerate(feature_names):
            shap_items.append(SHAP解释项(
                特征名称=fname,
                特征值=values[j] if not np.isnan(values[j]) else 0.0,
                SHAP值=float(shap_vals[j]),
                SHAP绝对值=float(abs(shap_vals[j]))
            ))

        shap_sorted = sorted(shap_items, key=lambda x: x.SHAP绝对值, reverse=True)
        top_factors = [f"{item.特征名称} (SHAP={item.SHAP值:+.4f})" for item in shap_sorted[:3]]

        return 文件预测响应(
            文件名=file.filename,
            提取特征数=matched_count,
            缺失特征数=len(missing),
            缺失特征=missing,
            风险概率=round(risk_prob, 6),
            风险判定=risk_result,
            模型基准值=round(base_val, 6),
            主要风险因素=top_factors
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"文件解析或预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
