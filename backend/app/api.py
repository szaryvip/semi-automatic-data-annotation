import random
import shutil
import time
import zipfile
import tempfile
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import AnswerPayload
from deepbackend.modules import SADATool


app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

sada_tool = None


@app.post("/upload")
def upload_files(files: list[UploadFile] = File(...)):
    global sada_tool
    for file in files:
        with open(f'media/unlabeled/{file.filename}', 'wb+') as destination:
            destination.write(file.file.read())
    sada_tool = SADATool()
    return 'Files uploaded successfully'


@app.delete("/delete-files")
def delete_files():
    if os.path.exists("media"):
        shutil.rmtree("media")
    if os.path.exists("annotated_data"):
        shutil.rmtree("annotated_data")
    os.mkdir("media")
    os.mkdir("annotated_data")
    os.mkdir("media/unlabeled")
    return 'Files deleted successfully'


@app.get("/prepare_vae")
def prepare_vae_model():
    global sada_tool
    if sada_tool is None:
        return "Firstly upload data!"
    sada_tool.prepare_vae()
    return "VAE model prepared!"


@app.get("/download_files")
def download_data():
    zip_filename = 'files.zip'

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip_file = os.path.join(temp_dir, zip_filename)

        with zipfile.ZipFile(temp_zip_file, 'w') as zip_file:
            for root, directories, files in os.walk('annotated_data/'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, 'annotated_data/')
                    zip_file.write(file_path, arcname=arcname)

        return StreamingResponse(
            open(temp_zip_file, 'rb'),
            media_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="{zip_filename}"'}
        )


@app.get("/get_images")
def get_images():
    global sada_tool
    if sada_tool is None:
        return
    sada_tool.select_to_manual_annotation()
    
    zip_filename = 'files.zip'

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip_file = os.path.join(temp_dir, zip_filename)

        with zipfile.ZipFile(temp_zip_file, 'w') as zip_file:
            for root, _, files in os.walk('media/'):
                for idx, file in enumerate(files):
                    if idx in sada_tool.selected_idx:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, 'media/')
                        zip_file.write(file_path, arcname=arcname)

        return StreamingResponse(
            open(temp_zip_file, 'rb'),
            media_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="{zip_filename}"'}
        )


@app.post("/submit_answers")
def submit_answers(answers: AnswerPayload):
    global sada_tool
    if sada_tool is not None:
        sada_tool.annotate_data(answers.answers)
        return JSONResponse({'responses': 'Answers received, data annotated!'})
    return JSONResponse({'responses': 'VAE is not prepared!'})
