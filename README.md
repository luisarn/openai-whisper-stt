# OpenAI API Compatible Whisper STT (mlx-whisper)

Offline speech recognition service based on [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) running the
[`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo) model on Apple Silicon,
providing OpenAI Whisper API compatible HTTP endpoints.

Adapted from the sibling project [openai-sensevoice-stt](../openai-sensevoice-stt) (FunASR/SenseVoice based),
keeping the same API surface, CLI style, and optional LLM post-processing (2-pass mode).

## Features

- ✅ **OpenAI API Compatible**: Supports `/v1/audio/transcriptions` endpoint
- ✅ **Multiple Response Formats**: JSON, Text, Verbose JSON, SRT, VTT (SRT/VTT use real per-segment timestamps from Whisper)
- ✅ **Multi-language Support**: ~100 languages via Whisper large-v3-turbo
- ✅ **Auto Language Detection**: Leave `language` unset for automatic detection
- ✅ **Apple Silicon Acceleration**: MLX runs on the Metal GPU — whisper-large-v3-turbo is fast and memory efficient on M-series Macs
- ✅ **In-memory Audio Processing**: uploads are decoded with ffmpeg to 16kHz mono float32 without touching disk (except a temporary buffer for ffmpeg)
- ⚠️ **Translation Mode**: `--task translate` is wired through, but whisper-large-v3-**turbo** was not trained for translation and its translate output is unreliable (it often just re-emits the source language). Use a non-turbo model (e.g. `mlx-community/whisper-large-v3`) if you need translation.
- ✅ **Word Timestamps**: optional `--word_timestamps` for precise alignment
- ✅ **LLM Post-Processing**: Optional 2-pass mode with LLM-based transcript correction for improved accuracy

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4) — **MLX does not work on Intel Macs or in Linux/Docker VMs** (no Metal GPU access)
- [ffmpeg](https://ffmpeg.org) installed (`brew install ffmpeg`)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
uv sync
```

The model (~1.6 GB) is downloaded automatically from HuggingFace on first run and cached in
`~/.cache/huggingface/hub`.

## Deployment

### Basic Startup

```bash
uv run whisper_http_server.py --port 8100
```

### Run in background

```bash
nohup uv run whisper_http_server.py --port 8100 > server.log 2>&1 &
```

### Custom Configuration

```bash
uv run whisper_http_server.py \
  --port 8100 \
  --model_dir mlx-community/whisper-large-v3-turbo \
  --language zh \
  --temperature 0.0
```

Use a local model directory (e.g. a quantized variant you downloaded yourself):

```bash
uv run whisper_http_server.py --model_dir ./models/whisper-large-v3-turbo
```

### LLM Post-Processing (2-Pass Mode)

Enable optional LLM-based transcript correction for improved accuracy:

```bash
uv run whisper_http_server.py --port 8100 --llm_correct
```

**Configuration:**

1. Create a `.env` file in the project root:
```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

2. (Optional) Customize the system prompt by creating `prompts/llm_correction_system.txt`:
```
You are an expert editor designed to post-process ASR transcripts.
Your task is to correct spelling, grammar, and punctuation...
```

**Notes:**
- If `prompts/llm_correction_system.txt` exists, it will be used as the system prompt
- Otherwise, a sensible default prompt is used
- Works with OpenAI-compatible APIs (OpenAI, Azure, local LLMs, etc.)
- The LLM correction applies to the `/v1/audio/transcriptions` endpoint

### Docker Deployment

Not supported: MLX requires the Metal GPU on macOS. Docker on Mac runs a Linux VM without
Metal access, so mlx-whisper cannot run in a container. Run it natively instead
(e.g. via `nohup` or a `launchd` user agent).

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--host` | str | `0.0.0.0` | Server listening address |
| `--port` | int | `8100` | Server port |
| `--model_dir` | str | `mlx-community/whisper-large-v3-turbo` | HF repo id or local model directory |
| `--language` | str | `None` | Default language code (`zh`, `en`, `ja`, ...); unset/auto = detect |
| `--task` | str | `transcribe` | `transcribe` or `translate` (to English) |
| `--temperature` | float | `0.0` | Default sampling temperature |
| `--word_timestamps` | flag | `False` | Enable word-level timestamps |
| `--certfile` / `--keyfile` | str | `None` | SSL cert/key files |
| `--temp_dir` | str | `temp_dir/` | Temp dir (created at startup) |
| `--llm_correct` | flag | `False` | Enable LLM-based transcript post-processing (requires `.env` config) |

## Usage

### 1. OpenAI API Compatible Endpoint

#### Basic Usage (JSON Format)

```bash
curl --request POST \
  --url http://localhost:8100/v1/audio/transcriptions \
  --header 'Content-Type: multipart/form-data' \
  --form file=@audio/example.wav \
  --form model=whisper-large-v3-turbo
```

**Response:**
```json
{"text":"欢迎大家来体验达摩院推出的语音识别模型。"}
```

#### Plain Text Format

```bash
curl --request POST \
  --url http://localhost:8100/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form model=whisper-large-v3-turbo \
  --form response_format=text
```

#### Verbose JSON Format (with segments)

```bash
curl --request POST \
  --url http://localhost:8100/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form model=whisper-large-v3-turbo \
  --form response_format=verbose_json
```

**Response:**
```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 10.0,
  "text": "欢迎大家来体验达摩院推出的语音识别模型。",
  "segments": [
    {"id": 0, "start": 0.0, "end": 10.0, "text": "欢迎大家来体验达摩院推出的语音识别模型。"}
  ]
}
```

#### SRT / VTT Subtitle Formats

```bash
curl --request POST \
  --url http://localhost:8100/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form response_format=srt
```

Unlike the SenseVoice server (which emits a single fake cue), this server renders real
multi-cue subtitles from Whisper's segment timestamps.

#### Translation to English

```bash
curl --request POST \
  --url http://localhost:8100/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form task=translate \
  --form response_format=text
```

### 2. Legacy Endpoint (Backward Compatible)

```bash
curl --request POST \
  --url http://localhost:8100/recognition \
  --header 'Content-Type: multipart/form-data' \
  --form audio=@audio/example.wav
```

**Response:**
```json
{
  "text": "欢迎大家来体验达摩院推出的语音识别模型。",
  "code": 0
}
```

### 3. Python Client

```bash
uv run whisper_http_client.py --port 8100 --audio_path audio/example.wav
```

### 4. OpenAI SDK

Because the API is OpenAI-compatible, the official SDK works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8100/v1", api_key="not-needed")

with open("audio/example.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=f,
        language="zh",
    )
print(result.text)
```

## API Reference

### POST /v1/audio/transcriptions

OpenAI Whisper API compatible transcription endpoint.

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | ✅ | - | Audio file (supports mp3, wav, m4a, webm, etc.) |
| `model` | string | ❌ | `whisper-large-v3-turbo` | Model name (informational — the server always uses `--model_dir`) |
| `language` | string | ❌ | auto-detect | Language code (ISO-639-1) |
| `prompt` | string | ❌ | - | Initial prompt to guide style/vocabulary |
| `response_format` | string | ❌ | `json` | Response format (`json`, `text`, `verbose_json`, `srt`, `vtt`) |
| `temperature` | float | ❌ | `0` | Sampling temperature |
| `task` | string | ❌ | `transcribe` | `transcribe` or `translate` (to English) |

### POST /recognition

Legacy endpoint, maintains backward compatibility with the FunASR HTTP client.

**Request Parameters:**
- `audio`: Audio file

**Response Format:**
```json
{
  "text": "Recognition result",
  "code": 0
}
```

## Performance Benchmarks

You can benchmark the server using the included `test_speed.py`:

```bash
# Start the server
uv run whisper_http_server.py --port 8100

# In another terminal, run the speed test
uv run test_speed.py audio/your_audio.wav --runs 10
```

On Apple Silicon, whisper-large-v3-turbo typically runs several times faster than realtime
(RTF well below 1.0), and noticeably faster than the SenseVoice CPU pipeline for long audio.

## Troubleshooting

### Port Already in Use

```bash
# Find process using the port
lsof -ti:8100

# Kill the process
lsof -ti:8100 | xargs kill -9
```

### Model Download Failed

The model downloads from HuggingFace on first run. If you are behind a firewall or mirror,
set the endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run whisper_http_server.py
```

### ffmpeg Error

Audio decoding is done with ffmpeg. Ensure it is installed:

```bash
brew install ffmpeg
```

### High Memory Usage

whisper-large-v3-turbo in fp16 uses ~1.6 GB. For a smaller footprint, use a quantized
variant, e.g.:

```bash
uv run whisper_http_server.py --model_dir mlx-community/whisper-large-v3-turbo-q4
```

## Credits

- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) / [MLX](https://github.com/ml-explore/mlx)
- [whisper-large-v3-turbo (mlx-community)](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [openai-sensevoice-stt](../openai-sensevoice-stt) — the project this server was adapted from
