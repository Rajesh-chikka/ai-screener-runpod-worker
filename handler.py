import os
import sys

print("=== HANDLER.PY STARTED ===", flush=True)
print(f"Python executable: {sys.executable}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)
print(f"Files: {os.listdir('.')}", flush=True)

import runpod

print("=== RUNPOD IMPORTED ===", flush=True)

from model_manager import model_manager

print("=== MODEL MANAGER IMPORTED ===", flush=True)


def handler(job):
    print("=== JOB RECEIVED ===", flush=True)

    return {
        "status": "diagnostic",
        "message": "handler.py is running",
    }


print("=== STARTING RUNPOD SERVERLESS ===", flush=True)

runpod.serverless.start(
    {
        "handler": handler
    }
)