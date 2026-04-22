from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import subprocess
import uuid
import os
import shutil

app = FastAPI()

MODEL_PATH = "/mnt/data0/Sougata/TTS/OmniVoice/OmniVoice/examples/exp/omnivoice_finetune/Indian_26_languages/checkpoint-5000"

BASE_DIR = "/mnt/data0/Sougata/TTS/API/OmniVoice/api_runtime"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "API running"}


# @app.post("/generate")
# async def generate_audio(
#     text: str = Form(...),
#     ref_text: str = Form(...),
#     instruct: str = Form("male, young adult, moderate pitch"),
#     duration: float = Form(1e-15),
#     guidance_scale: float = Form(9.0),
#     speed: float = Form(0.5),
#     denoise: bool = Form(False),
#     ref_audio: UploadFile = File(...),
#     preprocess_prompt: bool = Form(True),
#     postprocess_output: bool = Form(True)
# ):
#     try:
#         uid = str(uuid.uuid4())

#         # -----------------------------
#         # Save uploaded reference audio
#         # -----------------------------
#         ref_audio_path = os.path.join(UPLOAD_DIR, f"{uid}_{ref_audio.filename}")
#         with open(ref_audio_path, "wb") as f:
#             shutil.copyfileobj(ref_audio.file, f)

#         # -----------------------------
#         # Output path
#         # -----------------------------
#         output_path = os.path.join(OUTPUT_DIR, f"{uid}.wav")

#         # -----------------------------
#         # Build CLI command
#         # -----------------------------
#         cmd = [
#             "omnivoice-infer",
#             "--model", MODEL_PATH,
#             "--text", text,
#             "--ref_audio", ref_audio_path,
#             "--ref_text", ref_text,
#             "--instruct", instruct,
#             "--output", output_path,
#             "--duration", str(duration),
#             "--guidance_scale", str(guidance_scale),
#             "--speed", str(speed),
#             "--denoise", str(denoise),
#             "--preprocess_prompt", str(preprocess_prompt),
#             "--postprocess_output", str(postprocess_output)
#         ]

#         # -----------------------------
#         # Run inference
#         # -----------------------------
#         result = subprocess.run(cmd, capture_output=True, text=True)

#         if result.returncode != 0:
#             return {"error": result.stderr}

#         # -----------------------------
#         # Return generated audio
#         # -----------------------------
#         return FileResponse(output_path, media_type="audio/wav")

#     except Exception as e:
#         return {"error": str(e)}
@app.post("/generate")
async def generate_audio(
    text: str = Form(...),

    # Optional reference block
    ref_audio: UploadFile | None = File(None),
    ref_text: str | None = Form(None),

    # Optional params
    instruct: str | None = Form(None),
    duration: float | None = Form(None),
    guidance_scale: float | None = Form(None),
    speed: float | None = Form(None),

    denoise: bool | None = Form(None),
    preprocess_prompt: bool | None = Form(None),
    postprocess_output: bool | None = Form(None)
):
    try:
        uid = str(uuid.uuid4())

        output_path = os.path.join(OUTPUT_DIR, f"{uid}.wav")

        # -----------------------------
        # Base command
        # -----------------------------
        cmd = [
            "omnivoice-infer",
            "--model", MODEL_PATH,
            "--text", text,
            "--output", output_path,
        ]

        # -----------------------------
        # Handle reference block
        # -----------------------------
        if ref_audio and ref_text:
            ref_audio_path = os.path.join(UPLOAD_DIR, f"{uid}_{ref_audio.filename}")

            with open(ref_audio_path, "wb") as f:
                shutil.copyfileobj(ref_audio.file, f)

            cmd += ["--ref_audio", ref_audio_path]
            cmd += ["--ref_text", ref_text]

        elif ref_audio or ref_text:
            return {"error": "Both ref_audio and ref_text must be provided together"}

        # -----------------------------
        # Optional scalar arguments
        # -----------------------------
        if instruct:
            cmd += ["--instruct", instruct]

        if duration is not None:
            cmd += ["--duration", str(duration)]

        if guidance_scale is not None:
            cmd += ["--guidance_scale", str(guidance_scale)]

        if speed is not None:
            cmd += ["--speed", str(speed)]

        # -----------------------------
        # Boolean flags (only if False)
        # -----------------------------
        if denoise is False:
            cmd += ["--denoise", "False"]

        if preprocess_prompt is False:
            cmd += ["--preprocess_prompt", "False"]

        if postprocess_output is False:
            cmd += ["--postprocess_output", "False"]

        # -----------------------------
        # Run inference
        # -----------------------------
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return {"error": result.stderr}

        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        return {"error": str(e)}