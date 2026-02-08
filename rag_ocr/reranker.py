from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像（可选）

app = FastAPI()
model_name = "jinaai/jina-reranker-v2-base-multilingual"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.float16,  # 关键：用 dtype 替代 deprecated 的 torch_dtype
    device_map="auto"
)
model.eval()

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

@app.post("/rerank")
def rerank(request: RerankRequest):
    pairs = [[request.query, doc] for doc in request.documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(** inputs)
    scores = outputs.logits.squeeze().tolist()
    return [{"document": doc, "score": score} for doc, score in zip(request.documents, scores)]



# 若直接运行该脚本，自动启动服务
if __name__ == "__main__":
    import uvicorn
    # 关键参数：host设为0.0.0.0（允许外部访问），port指定端口
    # 可选参数：reload=True（开发环境热重载，生产环境关闭）
    uvicorn.run(
        "main:app",  # 模块名:app实例
        host="127.0.0.1",
        port=8020,
        reload=False,  # 生产环境禁用热重载
        workers=2  # 多进程（根据CPU核心数调整，提升并发）
    )