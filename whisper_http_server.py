import argparse
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

import mlx_whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-http-server")
logger.setLevel(logging.INFO)


def process_audio_bytes_ffmpeg(audio_bytes: bytes, suffix: str = "wav") -> np.ndarray:
    """
    使用 ffmpeg 將上傳的音頻 bytes 轉換為 16kHz 單聲道 float32 numpy array
    （透過臨時文件，ffmpeg-python 支援任意輸入格式）

    Args:
        audio_bytes: 原始音頻文件的 bytes
        suffix: 文件後綴名（如 wav, mp3, m4a）

    Returns:
        numpy array of audio samples (16kHz, mono, float32)
    """
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        pcm_bytes, _ = (
            ffmpeg.input(tmp.name, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )

    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_array


def transcribe_audio(
    audio_array: np.ndarray,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    task: str = "transcribe",
) -> dict:
    """
    使用 mlx-whisper 進行語音辨識

    Args:
        audio_array: 16kHz 單聲道 float32 numpy array
        language: 語言代碼（None 表示自動檢測）
        prompt: 初始提示詞（用於引導風格/詞彙）
        temperature: 採樣溫度
        task: transcribe 或 translate

    Returns:
        dict with "text", "language", "segments"
    """
    kwargs = {
        "path_or_hf_repo": args.model_dir,
        "temperature": temperature,
        "task": task,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": True,
        "verbose": False,
    }
    if language and language.lower() not in ("auto",):
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt

    result = mlx_whisper.transcribe(audio_array, **kwargs)
    return result


def llm_correction(text: str) -> str:
    """
    使用 LLM 進行文本校正

    Args:
        text: 需要校正的文本

    Returns:
        校正後的文本
    """
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in llm_correction: {e}")
        return text


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=8100, required=False, help="server port")
parser.add_argument(
    "--model_dir",
    type=str,
    default="mlx-community/whisper-large-v3-turbo",
    help="mlx-whisper model name (HF repo) or local directory path",
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
    help="Default language for ASR (None/auto for automatic detection, or zh, en, ja, ...)",
)
parser.add_argument(
    "--task",
    type=str,
    default="transcribe",
    choices=["transcribe", "translate"],
    help="transcribe (原語言) or translate (翻譯為英文)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Default sampling temperature",
)
parser.add_argument(
    "--word_timestamps",
    action="store_true",
    help="Enable word-level timestamps (slower, used for precise subtitles)",
)
parser.add_argument("--certfile", type=str, default=None, required=False, help="certfile for ssl")
parser.add_argument("--keyfile", type=str, default=None, required=False, help="keyfile for ssl")
parser.add_argument("--temp_dir", type=str, default="temp_dir/", required=False, help="temp dir")
parser.add_argument("--llm_correct", action="store_true", help="enable llm correction")
args = parser.parse_args()
logger.info("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    logger.info("%s: %s" % (arg, value))
logger.info("------------------------------------------------")

os.makedirs(args.temp_dir, exist_ok=True)

logger.info(f"loading model: {args.model_dir} (首次運行會自動從 HuggingFace 下載)")
# 預熱模型：觸發下載與載入，之後 mlx-whisper 會快取模型
_warmup = mlx_whisper.transcribe(
    np.zeros(16000, dtype=np.float32),  # 1 秒靜音
    path_or_hf_repo=args.model_dir,
    verbose=False,
)
logger.info("model loaded!")

client = None
LLM_SYSTEM_PROMPT = None
if args.llm_correct:
    from dotenv import load_dotenv
    from openai import OpenAI

    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))

    # 讀取 LLM 系統提示
    prompt_file = Path.cwd() / "prompts" / "llm_correction_system.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            LLM_SYSTEM_PROMPT = f.read()
        logger.info(f"已從 {prompt_file} 載入 LLM 系統提示")
    except Exception as e:
        logger.error(f"無法讀取 LLM 系統提示文件 {prompt_file}: {e}")
        logger.error("將使用預設的系統提示")
        LLM_SYSTEM_PROMPT = """
You are an expert editor designed to post-process ASR (Automatic Speech Recognition) transcripts.

Your task is to correct the provided text while strictly adhering to the following rules:

1. **Fix Errors:** Correct spelling, grammar, and punctuation mistakes.
2. **Contextual Correction:** Fix obvious phonetic mistranscriptions (homophones) based on the context.
3. **Formatting:** Restore proper capitalization for sentences and proper nouns.
4. **Preserve Meaning:** Do NOT change the original meaning, tone, or style of the speaker. Do NOT summarize or hallucinate information.
5. **Output Constraint:** Output ONLY the corrected text. Do not include any conversational fillers, introductions, or explanations (e.g., do not say "Here is the corrected text").
6. **Language:** Follow the original language of the transcript.
        """

    logger.info("2 Pass mode enable!")

app = FastAPI(title="MLX-Whisper STT")


# Helper functions for OpenAI API response formatting
def format_srt_time(seconds: Optional[float]) -> str:
    """將秒數轉換為 SRT 時間格式 (HH:MM:SS,mmm)"""
    if seconds is None:
        return "00:00:10,000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: Optional[float]) -> str:
    """將秒數轉換為 VTT 時間格式 (HH:MM:SS.mmm)"""
    if seconds is None:
        return "00:00:10.000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """將 whisper segments 格式化為 SRT 字幕（多條 cue，含真實時間戳）"""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_srt_time(seg.get("start"))
        end = format_srt_time(seg.get("end"))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def segments_to_vtt(segments: list) -> str:
    """將 whisper segments 格式化為 WebVTT 字幕（多條 cue，含真實時間戳）"""
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(segments, start=1):
        start = format_vtt_time(seg.get("start"))
        end = format_vtt_time(seg.get("end"))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_transcription_response(
    result: dict, response_format: str = "json", language: Optional[str] = None
):
    """
    將 whisper 辨識結果格式化為指定的回應格式

    Args:
        result: mlx_whisper.transcribe 的結果 dict（text, segments, language）
        response_format: 回應格式 (json, text, verbose_json, srt, vtt)
        language: 請求中指定的語言（可選）

    Returns:
        格式化後的回應
    """
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    detected_language = result.get("language") or language

    if response_format == "text":
        return text
    elif response_format == "json":
        return {"text": text}
    elif response_format == "verbose_json":
        duration = segments[-1]["end"] if segments else None
        return {
            "task": args.task,
            "language": detected_language,
            "duration": duration,
            "text": text,
            "segments": [
                {
                    "id": seg.get("id"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", "").strip(),
                }
                for seg in segments
            ],
        }
    elif response_format == "srt":
        return segments_to_srt(segments) if segments else f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"
    elif response_format == "vtt":
        return segments_to_vtt(segments) if segments else f"WEBVTT\n\n1\n00:00:00.000 --> 00:00:10.000\n{text}\n"
    else:
        # 預設返回 json
        return {"text": text}


@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="audio file")):
    """Legacy endpoint，保持與 funasr http server 向後兼容"""
    content = await audio.read()

    try:
        suffix = audio.filename.split(".")[-1] if audio.filename and "." in audio.filename else "wav"
        audio_array = process_audio_bytes_ffmpeg(content, suffix)
        result = await asyncio.to_thread(
            transcribe_audio, audio_array, language=args.language, task=args.task
        )
    except Exception as e:
        logger.error(f"读取音频文件发生错误，错误信息：{e}")
        return {"msg": "读取音频文件发生错误", "code": 1}

    text = (result.get("text") or "").strip()
    if not text:
        return {"text": "", "code": 0}

    ret = {"text": text, "code": 0}
    logger.info(f"識別結果：{ret}")
    return ret


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(..., description="audio file"),
    model_name: Optional[str] = Form("whisper-large-v3-turbo", alias="model", description="model to use"),
    language: Optional[str] = Form(None, description="language code (ISO-639-1)"),
    prompt: Optional[str] = Form(None, description="optional prompt"),
    response_format: Optional[str] = Form("json", description="response format"),
    temperature: Optional[float] = Form(0, description="sampling temperature"),
    task: Optional[str] = Form(None, description="transcribe or translate"),
):
    """
    OpenAI-compatible audio transcription endpoint

    Compatible with OpenAI's /v1/audio/transcriptions API
    Supports response formats: json, text, verbose_json, srt, vtt
    """
    content = await file.read()

    try:
        suffix = file.filename.split(".")[-1] if file.filename and "." in file.filename else "wav"
        audio_array = process_audio_bytes_ffmpeg(content, suffix)
    except Exception as e:
        logger.error(f"音頻文件處理錯誤：{e}")
        return {
            "error": {"message": "Audio file processing failed", "type": "invalid_request_error"}
        }

    try:
        # mlx-whisper 推理為同步阻塞操作，放入線程池避免阻塞事件循環
        result = await asyncio.to_thread(
            transcribe_audio,
            audio_array,
            language=language or args.language,
            prompt=prompt,
            temperature=temperature if temperature is not None else args.temperature,
            task=task or args.task,
        )

        text = (result.get("text") or "").strip()
        if args.llm_correct and text:
            text = await asyncio.to_thread(llm_correction, text)
            result = dict(result, text=text)

        # 格式化回應
        formatted_response = format_transcription_response(
            result=result,
            response_format=response_format,
            language=language,
        )

        logger.info(f"OpenAI API 辨識結果：{text[:100]}...")

        # 根據回應格式設定 Content-Type
        if response_format in ("text", "srt", "vtt"):
            return Response(content=formatted_response, media_type="text/plain")
        return formatted_response

    except Exception as e:
        logger.error(f"辨識過程發生錯誤：{e}")
        return {"error": {"message": str(e), "type": "server_error"}}


if __name__ == "__main__":
    uvicorn.run(
        app, host=args.host, port=args.port, ssl_keyfile=args.keyfile, ssl_certfile=args.certfile
    )
