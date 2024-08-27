from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import importlib.util
import sys
from pathlib import Path
import os
import PyPDF2
# import docx

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Import PredictionPipeline using a file path
prediction_file_path = Path("src/textSummarizer/pipeline/prediction.py")
spec = importlib.util.spec_from_file_location("prediction", prediction_file_path)
prediction_module = importlib.util.module_from_spec(spec)
sys.modules["prediction"] = prediction_module
spec.loader.exec_module(prediction_module)

PredictionPipeline = prediction_module.PredictionPipeline

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
# def extract_text_from_docx(file):
#     doc = docx.Document(file)
#     text = ''
#     for paragraph in doc.paragraphs:
#         text += paragraph.text + '\n'
#     return text

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize_text(
    request: Request, 
    text: str = Form(None), 
    summary_length: str = Form("medium"), 
    file: UploadFile = File(None)
):
    try:
        # Initialize an empty text variable
        extracted_text = ""

        # Check if a file was uploaded
        if file:
            if file.filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file.file)
            # Uncomment if using docx
            # elif file.filename.endswith(".docx"):
            #     extracted_text = extract_text_from_docx(file.file)
            else:
                return templates.TemplateResponse(
                    "index.html", 
                    {
                        "request": request, 
                        "summary": "Unsupported file type. Please upload a PDF or DOCX file.",
                        "text": "",
                        "summary_length": summary_length
                    }
                )
        else:
            # If text input was provided instead of a file
            extracted_text = text

        # If no text was provided and no file was uploaded, return an error
        if not extracted_text:
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request, 
                    "summary": "No text provided. Please paste text or upload a file.",
                    "text": "",
                    "summary_length": summary_length
                }
            )

        # Use PredictionPipeline to summarize the text
        obj = PredictionPipeline()
        summarized_text = obj.predict(extracted_text, summary_length=summary_length)
        summary_length_count = len(summarized_text)

    except Exception as e:
        summarized_text = f"Error: {str(e)}"
        summary_length_count = 0

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "summary": summarized_text, 
            "text": extracted_text, 
            "summary_length": summary_length,
            "summary_length_count": summary_length_count  # Pass the length of the summary to the template
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)