import sys
import json
import asyncio
import yaml
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastai.text.all import load_learner


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


config = load_config()
model_path = config.get("model_path")
tokens = config.get("tokens", [])
record_path = config.get("record_path", "record.xls")
host = config.get("host", "127.0.0.1")
port = config.get("port", 3001)
record_enabled = config.get("record_enabled", False)
max_content_length = config.get("max_content_length", 3000)

if not model_path or not tokens:
    print(json.dumps({'error': 'Model path or tokens missing in config.'}))
    sys.exit(1)


def load_model(model_path: str):
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        print(f"Model loading failed: {str(e)}", file=sys.stderr)
        return None


learn = load_model(model_path)

app = FastAPI()

security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in tokens:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing token"
        )


def validate_content_length(content):
    return content[:max_content_length] if len(content) > max_content_length else content


def is_invisible(content: str) -> bool:
    return not any(char.isprintable() for char in content)


def record_result(note: str, score: float):
    if record_enabled:
        try:
            with open(record_path, 'a') as f:
                f.write(f"{note}\t{score}\n")
        except Exception as e:
            print(f"Record failed: {str(e)}", file=sys.stderr)


async def predict(content: str):
    truncated_content = validate_content_length(content)
    prediction = await asyncio.to_thread(learn.predict, truncated_content)
    label = prediction[0]
    score = float("{:.4f}".format(prediction[2][prediction[1]].item()))
    return label, score


if not learn:
    print(json.dumps({'error': 'Model loading failed at startup.'}))
    sys.exit(1)


@app.post("/predict/")
async def predict_endpoint(request: Request, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        data = await request.json()
        note = data.get("note", "")

        if not note:
            raise HTTPException(status_code=400, detail="Note is required")

        if len(note) < 3:
            raise HTTPException(status_code=400, detail="Note is too short")

        if is_invisible(note):
            raise HTTPException(status_code=400, detail="Note contains only invisible characters")

        label, score = await predict(note)

        record_result(note, score)

        return JSONResponse(content={"label": label, "score": score})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
