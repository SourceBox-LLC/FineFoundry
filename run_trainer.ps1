$WORKDIR = (Resolve-Path .).Path
docker run `
  --gpus all `
  --name unsloth-train `
  --shm-size=8g --ipc=host `
  -e PYTHONUNBUFFERED=1 `
  -v "${WORKDIR}:/workspace/work" `
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
  -w /workspace/work `
  unsloth/unsloth:latest `
  python trainer.py