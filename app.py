from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import importlib.util
import sys
from pathlib import Path

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Import PredictionPipeline using a file path
prediction_file_path = Path("src/textSummarizer/pipeline/prediction.py")
spec = importlib.util.spec_from_file_location("prediction", prediction_file_path)
prediction_module = importlib.util.module_from_spec(spec)
sys.modules["prediction"] = prediction_module
spec.loader.exec_module(prediction_module)

PredictionPipeline = prediction_module.PredictionPipeline

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize_text(request: Request, text: str = Form(...), summary_length: str = Form("medium")):
    try:
        obj = PredictionPipeline()
        summarized_text = obj.predict(text, summary_length=summary_length)
    except Exception as e:
        summarized_text = f"Error: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "summary": summarized_text, "text": text, "summary_length": summary_length})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)