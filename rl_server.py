import json
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime

from optimal_morphology_rl_pipeline.rl_pipeline import (
    build_test_cmd,
    build_train_cmd,
    cache_policy,
    clear_vlearn_convex_cache,
    create_videos,
    find_cached_policy,
    generate_vsim_from_json,
    run_cmd,
    zip_run_artifacts,
)

DATA_DIR = Path("/workspace/data")
SERVER_DATA_DIR = Path("/workspace/data/server")

app = FastAPI()
semaphore = asyncio.Semaphore(4)  # Max 4 concurrent heavy processes
background_jobs: dict[str, asyncio.Task] = {}


async def run_rl(timestamp: str, object: str, cached_policy: Optional[Path]) -> None:
    vsim_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.vsim"
    folder_path = SERVER_DATA_DIR / timestamp

    cmd = build_train_cmd(
        folder_path,
        vsim_path,
        object,
        cached_policy,
        max_epochs=750,
        num_envs=4096,
    )
    log_path = folder_path / f"{vsim_path.stem}_train_output.log"

    # Run the blocking subprocess in a thread pool.
    await asyncio.to_thread(run_cmd, cmd, log_path=log_path)


async def check_for_cached_policy(object: str, num_dof: int) -> Optional[Path]:
    return find_cached_policy(
        object,
        num_dof,
        excluded_objects=("cube", "button_difficult", "tomato_extreme"),
    )


async def cache_trained_policy(timestamp: str, object: str, num_dof: int) -> Optional[Path]:
    folder_path = SERVER_DATA_DIR / timestamp
    return cache_policy(
        object,
        num_dof,
        folder_path,
        skip_if_exists=True,
    )


async def test_rl(timestamp: str, object: str) -> None:
    folder_path = SERVER_DATA_DIR / timestamp
    vsim_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.vsim"

    cmd = build_test_cmd(folder_path, vsim_path, object)
    log_path = folder_path / f"{vsim_path.stem}_test_output.log"

    await asyncio.to_thread(run_cmd, cmd, log_path=log_path)


async def generate_vsim(timestamp: str) -> int:
    json_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.json"
    vsim_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.vsim"

    print(f"Loading parameters from {json_path}")
    num_tendons = await asyncio.to_thread(generate_vsim_from_json, json_path, vsim_path)
    print(f"Generated VSIM with {num_tendons} tendons: {vsim_path}")
    return num_tendons


async def zip_results(timestamp: str) -> Path:
    folder_path = SERVER_DATA_DIR / timestamp
    zip_path = await asyncio.to_thread(zip_run_artifacts, folder_path, timestamp)
    return zip_path


async def process_json(timestamp, file: UploadFile = File(...)) -> Path:
    json_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.json"
    content = await file.read()
    with open(json_path, "wb") as f:
        f.write(content)
    return json_path


async def create_folder_for_timestamp(timestamp: str):
    folder = SERVER_DATA_DIR / timestamp
    folder.mkdir(parents=True, exist_ok=True)


def _cleanup_background_job(task: asyncio.Task):
    timestamp = task.get_name()
    background_jobs.pop(timestamp, None)
    try:
        task.result()
    except Exception as exc:
        print(f"Background job {timestamp} failed: {exc}")


async def run_pipeline(timestamp: str, object: str):
    clear_vlearn_convex_cache()
    num_dof = await generate_vsim(timestamp)
    cached_policy = await check_for_cached_policy(object, num_dof)
    await run_rl(timestamp, object, cached_policy)
    await cache_trained_policy(timestamp, object, num_dof)
    await test_rl(timestamp, object)
    await asyncio.to_thread(create_videos, SERVER_DATA_DIR / timestamp, timestamp)
    await zip_results(timestamp)
    print(f"Background job {timestamp} finished")


@app.get("/check")
async def check_request(timestamp: str):
    task = background_jobs.get(timestamp)
    if task is None:
        zip_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.zip"
        if zip_path.exists():
            return {"timestamp": timestamp, "status": "completed"}
        return {"timestamp": timestamp, "status": "not_found"}

    if task.cancelled():
        return {"timestamp": timestamp, "status": "failed"}

    if task.done():
        try:
            task.result()
        except Exception as exc:
            return {"timestamp": timestamp, "status": "failed", "error": str(exc)}
        return {"timestamp": timestamp, "status": "completed"}

    return {"timestamp": timestamp, "status": "running"}


async def process(file: UploadFile = File(...), object: str = "tomato"):
    """Receive JSON file, process locally, return results"""

    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(400, "Only .json files are allowed")

    try:
        # Read and parse the uploaded JSON

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        suffix = 0
        timestamp = f"{timestamp}_{suffix:02d}"
        while timestamp in background_jobs:
            suffix += 1
            timestamp = f"{timestamp}_{suffix:02d}"
        await create_folder_for_timestamp(timestamp)
        await process_json(timestamp, file)
        task = asyncio.create_task(run_pipeline(timestamp, object), name=timestamp)
        background_jobs[timestamp] = task
        task.add_done_callback(_cleanup_background_job)
        return {"status": "started", "timestamp": timestamp}

    except json.JSONDecodeError:
        raise HTTPException(400, f"Timestamp: {timestamp} - Invalid JSON file uploaded: {file.filename}")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")


@app.post("/run/tomato")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="tomato")

@app.post("/run/tomato_extreme")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="tomato_extreme")

@app.post("/run/drawer")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="drawer")

@app.post("/run/button")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="button")

@app.post("/run/button_difficult")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="button_difficult")

@app.post("/run/cube")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="cube")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "server": "local"}


@app.get("/download")
async def download_request(timestamp: str):
    zip_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.zip"
    if not zip_path.exists():
        raise HTTPException(404, f"No zip file found for timestamp: {timestamp}")

    return FileResponse(path=zip_path, filename=f"{timestamp}.zip", media_type="application/zip")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
