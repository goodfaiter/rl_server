import json
import asyncio
import shutil
from pathlib import Path
from typing import Tuple
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime

DATA_DIR = Path("/workspace/data")
SERVER_DATA_DIR = Path("/workspace/data/server")

app = FastAPI()
semaphore = asyncio.Semaphore(4)  # Max 4 concurrent heavy processes
background_jobs: dict[str, asyncio.Task] = {}


async def run_cmd(cmd: list) -> Tuple[str, str]:
    """Run a command asynchronously and capture stdout/stderr."""
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=None)
    stdout, stderr = await proc.communicate()
    return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


async def run_rl(timestamp: str, object: str, cached_policy: Path) -> Tuple[Path, Path]:
    vsim_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.vsim"
    folder_path = SERVER_DATA_DIR / timestamp

    max_epochs = 750
    if cached_policy:
        max_epochs = 100  # Reduce epochs if using cached policy

    cmd = [
        "uv",
        "run",
        "python3",
        "/workspace/vlearn/train/rl_games_train.py",
        "hand",
        "train",
        str(cached_policy) if cached_policy else "",
        "--headless",
        "True",
        "--vsim_path",
        str(vsim_path),
        "--experiment_name",
        str(folder_path),
        "--max_epochs",
        str(max_epochs),
        "--num_envs",
        "4096",
        "--object",
        object,
    ]

    # Wait for completion and capture output
    stdout_str, stderr_str = await run_cmd(cmd)

    # write to log files for debugging
    train_output_name = f"{vsim_path.stem}_train_output.log"
    train_output_path = folder_path / train_output_name
    with open(train_output_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(stdout_str)
        f.write("\n\nSTDERR:\n")
        f.write(stderr_str)

async def check_for_cached_policy(timestamp: str, object: str, num_dof: int) -> Path:
    """Check if a cached policy exists for the given object and number of DOF."""
    cache_dir = DATA_DIR / "cache" / object
    matches = list(cache_dir.glob(f"{num_dof}_*_cache.pth"))
    return matches[0] if matches else None

async def cache_policy(timestamp: str, object: str, num_dof: int) -> Path:
    """Cache the trained policy for later use."""
    if await check_for_cached_policy(timestamp, object, num_dof) is not None:
        return
    folder_path = SERVER_DATA_DIR / timestamp
    policy_path = folder_path / "nn/hand_object.pth"
    cache_path = DATA_DIR / "cache" / object / f"{num_dof}_{timestamp}_cache.pth"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_path, cache_path)
    return cache_path

async def test_rl(timestamp: str, object: str) -> Tuple[Path, Path]:
    folder_path = SERVER_DATA_DIR / timestamp
    policy_path = folder_path / "nn/hand_object.pth"
    vsim_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.vsim"

    cmd = [
        "uv",
        "run",
        "python3",
        "/workspace/vlearn/train/rl_games_train.py",
        "hand",
        "play",
        str(policy_path),
        "--headless",
        "True",
        "--vsim_path",
        str(vsim_path),
        "--games_num",
        "1",
        "--max_episode_length",
        "240",
        "--num_envs",
        "1",
        "--record_output_path",
        str(folder_path),
        "--object",
        object,
    ]

    # Wait for completion and capture output
    stdout_str, stderr_str = await run_cmd(cmd)

    # write to log files for debugging
    test_output_name = f"{vsim_path.stem}_test_output.log"
    test_output_path = folder_path / test_output_name
    with open(test_output_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(stdout_str)
        f.write("\n\nSTDERR:\n")
        f.write(stderr_str)

async def png_to_video(folder_path: Path, timestamp: str):
    cmd = [
        "ffmpeg",
        "-framerate",
        "60",
        "-i",
        str(folder_path / "step_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(folder_path / f"{timestamp}.mp4"),
    ]
    stdout_str, stderr_str = await run_cmd(cmd)

    # write to log files for debugging
    test_output_name = f"{timestamp}_video.log"
    test_output_path = folder_path / test_output_name
    with open(test_output_path, "w") as f:
        f.write("STDOUT:\n")
        f.write(stdout_str)
        f.write("\n\nSTDERR:\n")
        f.write(stderr_str)

async def create_video(timestamp: str):
    # run ffmpeg -framerate 60 -i step_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4
    cameras = ["camera_0", "camera_1"]
    for camera in cameras:
        folder_path = SERVER_DATA_DIR / timestamp / f"table_with_camera/{camera}"
        await png_to_video(folder_path, timestamp)

async def generate_vsim(timestamp: str) -> int:
    from oph_to_urdf import load_params, generate_hand_urdf, urdf_to_text

    json_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.json"

    print(f"Loading parameters from {json_path}")
    params = load_params(json_path)

    nf = params["n_fingers"]
    nt = params["n_thumbs"]
    print(f"Generating URDF: {nf} finger(s), {nt} thumb(s)")

    urdf = generate_hand_urdf(params)

    urdf_text = urdf_to_text(urdf, pretty_print=True)

    vsim_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.vsim"
    vsim_path.write_text(urdf_text)
    print(f"URDF written to {vsim_path}")

    n_links = len(urdf.robot.links)
    n_joints = len(urdf.robot.joints)
    n_rev = sum(1 for j in urdf.robot.joints if j.type == "revolute")
    print(f"  Links: {n_links}  |  Joints: {n_joints} ({n_rev} revolute) | Tendons: {len(urdf._tendons)}")

    return len(urdf._tendons)


async def zip_results(timestamp: str) -> FileResponse:
    zip_path = SERVER_DATA_DIR / timestamp / f"{timestamp}.zip"
    reward_log_path = SERVER_DATA_DIR / timestamp / f"{timestamp}_reward_log.txt"
    cameras = ["camera_0", "camera_1"]
    video_output_paths = [SERVER_DATA_DIR / timestamp / f"table_with_camera/{camera}" / f"{timestamp}.mp4" for camera in cameras]
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(reward_log_path, reward_log_path.name)
        for i, video_output_path in enumerate(video_output_paths):
            zipf.write(video_output_path, video_output_path.name + f"_{i:02d}.mp4")


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
    num_dof = await generate_vsim(timestamp)
    cached_policy = await check_for_cached_policy(timestamp, object, num_dof)
    await run_rl(timestamp, object, cached_policy)
    await cache_policy(timestamp, object, num_dof)
    await test_rl(timestamp, object)
    await create_video(timestamp)
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

@app.post("/run/drawer")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="drawer")

@app.post("/run/button")
async def process_request(file: UploadFile = File(...)):
    return await process(file, object="button")

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
