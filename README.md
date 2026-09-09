# RunPod Vision Inference Server

This isolated FastAPI scaffold is the future GPU inference service for AI-Screener. The existing backend can call it over HTTPS using the OpenAI-compatible `POST /v1/chat/completions` endpoint.

No model runtime is implemented yet. Requests for recognized model IDs return a model-not-loaded response instead of fabricated clinical output.

## Planned Backends

- `medgemma` for a configured MedGemma deployment
- `hf:<repository>` for Hugging Face multimodal models
- `ollama:<model-name>` for models served through Ollama

The public endpoint will remain unchanged as these backends are added to `ModelManager`.

## Environment

```dotenv
RUNPOD_SERVER_API_KEY=<strong-random-token>
RUNPOD_MAX_IMAGES=8
RUNPOD_MAX_IMAGE_MB=8
RUNPOD_DEVICE=cuda

# Future backend-specific variables
HF_TOKEN=<optional-hugging-face-token>
OLLAMA_HOST=<optional-ollama-url>
```

When `RUNPOD_SERVER_API_KEY` is non-empty, chat completion requests must include `Authorization: Bearer <token>`. The health endpoint is public.

## Future Startup

From this directory on RunPod:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected shape:

```json
{"status":"ok","device":"cuda","loaded_models":[]}
```

## Chat Completion Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medgemma",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64-data>"}},
        {"type": "text", "text": "<clinical-prompt>"}
      ]
    }],
    "max_tokens": 2048,
    "temperature": 0.1
  }'
```

After a model backend is implemented, a successful response has this shape:

```json
{
  "id": "chatcmpl-<request-id>",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "<model-json-text>"
    },
    "finish_reason": "stop"
  }]
}
```

Configure AI-Screener later with:

```dotenv
RUNPOD_VISION_ENABLED=true
RUNPOD_VISION_API_URL=https://<runpod-host>/v1/chat/completions
RUNPOD_VISION_API_KEY=<same-token>
RUNPOD_VISION_MODEL=<model-id>
```
