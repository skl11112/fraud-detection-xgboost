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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
