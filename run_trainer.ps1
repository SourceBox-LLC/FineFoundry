$WORKDIR = (Resolve-Path .).Path

# Configurable resource limits
$NAME    = "unsloth-train"
$CPUS    = 1
$CPUSET  = "0"
$MEMORY  = "8g"        # Adjust if you can spare more RAM
$MEMSWAP = "16g"       # Total memory+swap inside container; set > $MEMORY

docker run `
  --gpus all `
  --name $NAME `
  --rm `
  --entrypoint "" `
  --cpus $CPUS `
  --cpuset-cpus $CPUSET `
  --memory $MEMORY `
  --memory-swap $MEMSWAP `
  --shm-size=8g --ipc=host `
  -e PYTHONUNBUFFERED=1 `
  -v "${WORKDIR}:/workspace/work" `
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
  -w /workspace/work `
  unsloth/unsloth:latest `
  python trainer.py