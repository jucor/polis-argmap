import sys
import os
import datetime

embedModel = None
languageModel = None


def getTorchDeviceVersion():
    import torch
    versions = f"""\
Python: {sys.version}
PyTorch: {torch.__version__}"""

    if torch.cuda.is_available():
        versions += f"""
CUDA: {torch.version.cuda}
CUDNN: {torch.backends.cudnn.version()}
Device: {torch.cuda.get_device_name(0)}"""
    else:
        versions += f"""
No CUDA support. Using CPU."""
    return versions


def printTorchDeviceVersion():
    print(getTorchDeviceVersion())


def getDevice():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def getCUDAMemory():
    import torch
    if not torch.cuda.is_available():
        return 0, 0, 0
        
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])
    total_memory = sum([torch.cuda.mem_get_info(i)[1]
                       for i in range(torch.cuda.device_count())])
    allocated_memory = sum([torch.cuda.memory_allocated(i)
                           for i in range(torch.cuda.device_count())])

    return free_memory, allocated_memory, total_memory


def printCUDAMemory():
    free_memory, allocated_memory, total_memory = getCUDAMemory()
    if total_memory == 0:
        print("Running on CPU - no CUDA memory to report", flush=True, file=sys.stderr)
        return
        
    free_memory = round(free_memory/1024**3, 1)
    allocated_memory = round(allocated_memory/1024**3, 1)
    total_memory = round(total_memory/1024**3, 1)
    print(f"CUDA Memory: {free_memory} GB free, {allocated_memory} GB allocated, {total_memory} GB total",
          flush=True, file=sys.stderr)


def ensureCUDAMemory(required_memory_gb):
    import torch
    if not torch.cuda.is_available():
        print("Warning: Running on CPU, CUDA memory check skipped", file=sys.stderr)
        return True
        
    required_memory = required_memory_gb * 1024**3
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])

    if free_memory >= required_memory:
        return True

    raise Exception(
        f"Insufficient CUDA memory: {round(free_memory/1024**3,1)} GB free, {required_memory_gb} GB required")


def loadLanguageModel():
    global languageModel

    if languageModel is not None:
        return languageModel

    import torch
    from guidance import models

    MODEL_ID = os.getenv("MODEL_ID")
    MODEL_REVISION = os.getenv("MODEL_REVISION")

    if MODEL_ID is None:
        raise Exception(
            "Required: HuggingFace Model ID using MODEL_ID environment variable")

    device = getDevice()
    if device == "cuda":
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        MODEL_MINIMUM_MEMORY_GB = os.getenv("MODEL_MINIMUM_MEMORY_GB")
        if MODEL_MINIMUM_MEMORY_GB is not None:
            ensureCUDAMemory(int(MODEL_MINIMUM_MEMORY_GB))

    print(f"{datetime.datetime.now()} Initializing language model: {MODEL_ID} on {device}...")
    if MODEL_REVISION:
        print(f"Model Revision: {MODEL_REVISION}")

    device_map = "auto" if device == "cuda" else None
    languageModel = models.Transformers(
        model=MODEL_ID,
        device=device,
        **({'device_map': device_map} if device_map else {}),
        **({'revision': MODEL_REVISION} if MODEL_REVISION else {})
    )

    print(f"{datetime.datetime.now()} Language model initialized.")
    printCUDAMemory()

    return languageModel


def loadEmbeddingModel():
    import torch

    global embedModel

    if embedModel is not None:
        return embedModel

    EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")

    if EMBED_MODEL_ID is None:
        raise Exception(
            "Required: SentenceTransformer Model ID using EMBED_MODEL_ID environment variable")

    device = getDevice()
    if device == "cuda":
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        EMBED_MODEL_MINIMUM_MEMORY_GB = os.getenv("EMBED_MODEL_MINIMUM_MEMORY_GB")
        if EMBED_MODEL_MINIMUM_MEMORY_GB is not None:
            ensureCUDAMemory(int(EMBED_MODEL_MINIMUM_MEMORY_GB))

    print(f"{datetime.datetime.now()} Initializing embedding model: {EMBED_MODEL_ID} on {device}...")

    from sentence_transformers import SentenceTransformer
    embedModel = SentenceTransformer(EMBED_MODEL_ID, device=device)

    print(f"{datetime.datetime.now()} Embedding model initialized.")
    printCUDAMemory()

    return embedModel


def unloadEmbeddingModel():
    import torch
    global embedModel
    embedModel = None
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
