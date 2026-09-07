#!/usr/bin/env bash
# Cantonese STT test script — exercises the /v1/audio/transcriptions endpoint
# with Cantonese audio (audio/cantonese.wav, generated via macOS `say -v Sinji`).
#
# Usage:
#   ./test_cantonese.sh              # run all tests against localhost:8100
#   ./test_cantonese.sh 8100         # explicit port
#   HOST=example.com PORT=443 SCHEME=https ./test_cantonese.sh
#
# Requires: curl, and the server running:
#   uv run whisper_http_server.py --port 8100

set -euo pipefail

SCHEME="${SCHEME:-http}"
HOST="${HOST:-localhost}"
PORT="${PORT:-${1:-8100}}"
AUDIO="${AUDIO:-audio/cantonese.wav}"
URL="${SCHEME}://${HOST}:${PORT}/v1/audio/transcriptions"

cd "$(dirname "$0")"

if [[ ! -f "$AUDIO" ]]; then
  echo "Audio file not found: $AUDIO"
  echo "Regenerate it with:"
  echo "  say -v Sinji -o /tmp/yue.aiff '大家好，歡迎嚟到測試粵語語音辨識系統。'"
  echo "  ffmpeg -y -i /tmp/yue.aiff -ar 16000 -ac 1 audio/cantonese.wav"
  exit 1
fi

# check server is up
if ! curl -s -o /dev/null --max-time 3 "${SCHEME}://${HOST}:${PORT}/docs"; then
  echo "Server not reachable at ${SCHEME}://${HOST}:${PORT} — start it with:"
  echo "  uv run whisper_http_server.py --port ${PORT}"
  exit 1
fi

pass=0; fail=0
check() { # $1=label $2=expected substring $3=actual
  if [[ "$3" == *"$2"* ]]; then
    echo "  ✓ $1"; pass=$((pass+1))
  else
    echo "  ✗ $1 — expected substring '$2' not found"; fail=$((fail+1))
  fi
}

post() { # extra -F args after the file
  curl -s --max-time 120 -X POST "$URL" -F "file=@${AUDIO}" "$@"
}

echo "=== Cantonese STT tests — ${URL} ==="
echo "Audio: ${AUDIO} ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO" 2>/dev/null || echo '?')s)"
echo

echo "1. language=yue, json"
r=$(post -F language=yue -F response_format=json)
echo "   $r"
check "returns text field" '"text"' "$r"

echo "2. auto-detect language, json"
r=$(post -F response_format=json)
echo "   $r"
check "returns text field" '"text"' "$r"

echo "3. language=yue, plain text"
r=$(post -F language=yue -F response_format=text)
echo "   $r"
check "non-empty text" "大家好" "$r"

echo "4. language=yue, verbose_json"
r=$(post -F language=yue -F response_format=verbose_json)
echo "   ${r:0:300}"
check "has language field" '"language"' "$r"
check "has segments" '"segments"' "$r"

echo "5. language=yue, srt"
r=$(post -F language=yue -F response_format=srt)
echo "   $(echo "$r" | head -3 | tr '\n' '|')"
check "srt cue marker" '-->' "$r"

echo "6. language=yue, vtt"
r=$(post -F language=yue -F response_format=vtt)
echo "   $(echo "$r" | head -4 | tr '\n' '|')"
check "vtt header" 'WEBVTT' "$r"

echo "7. with initial prompt (guide vocabulary)"
r=$(post -F language=yue -F 'prompt=這是一段粵語廣東話測試。')
echo "   $r"
check "returns text field" '"text"' "$r"

echo "8. legacy /recognition endpoint"
r=$(curl -s --max-time 120 -X POST "${SCHEME}://${HOST}:${PORT}/recognition" -F "audio=@${AUDIO}")
echo "   $r"
check "code 0" '"code":0' "$r"

echo
echo "=== Results: ${pass} passed, ${fail} failed ==="
[[ $fail -eq 0 ]]
