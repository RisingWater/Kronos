import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import sys
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

app = FastAPI(title='Kronos', version='1.0')

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Kronos API",
        "version": "1.0", 
    }

class PredictData(BaseModel):
    timestamps: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictRequest(BaseModel):
    predict_len: int #'daily is valid' 'minute always 1'
    data: List[PredictData]

class PredictResponse(BaseModel):
    prediction: List[PredictData]

from typing import List
import pandas as pd
from pydantic import BaseModel
from fastapi import HTTPException

class PredictData(BaseModel):
    timestamps: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictRequest(BaseModel):
    predict_len: int  # 预测的长度
    data: List[PredictData]  # 历史数据

class PredictResponse(BaseModel):
    prediction: List[PredictData]

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """预测接口"""

    request_data = [item.model_dump() for item in request.data]
    
    # 转换为 DataFrame
    df = pd.DataFrame(request_data)
    
    # 转换时间戳格式
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # 排序确保时间顺序
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    # lookback 等于请求中数据的长度
    lookback = len(df)

    if lookback < 200:  # 如果没有数据，返回错误
        raise HTTPException(status_code=400, detail="data too less")
    
    # 预测长度从请求中获取
    pred_len = request.predict_len
    
    # 提取 x_df（使用全部传入数据）
    x_df = df[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    
    # 提取 x_timestamp（全部时间戳）
    x_timestamp = df['timestamps'].reset_index(drop=True)
    
    # 生成 y_timestamp（预测时间段的时间戳）
    last_timestamp = df['timestamps'].iloc[-1]
    
    time_interval = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]

    # 生成 DatetimeIndex
    y_timestamp_index = pd.date_range(
        start=last_timestamp + time_interval,
        periods=pred_len,
        freq=time_interval
    )
    
    # 转换为 Series，保持与 x_timestamp 类型一致
    y_timestamp = pd.Series(y_timestamp_index, name='timestamps')
    
    # 调用预测函数
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )
    
    # 构建响应
    predictions = []
    for i, row in pred_df.iterrows():
        timestamp = row['timestamps']
        # 转换时间戳为字符串
        if hasattr(timestamp, 'isoformat'):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)
        
        predictions.append(PredictData(
            timestamps=timestamp_str,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row.get('volume', 0.0))  # 如果volume不存在，使用0.0
        ))
    
    return PredictResponse(prediction=predictions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6030)