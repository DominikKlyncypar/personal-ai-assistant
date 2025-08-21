from fastapi import FastAPI
from pydantic import BaseModel

class EchoIn(BaseModel):
    message: str

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/echo")
def echo(msg: str = "hello"):
    return {"echo": msg}

@app.post("/echo")
def echo_post(payload: EchoIn):
    return {"echo": payload.message}