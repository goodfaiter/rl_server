# RL Server for testing morphological adaptation

To start:
```
cloudflared tunnel run --token CLOUDFLARE_TOKEN
uv run python3 main.py
```

To access:
```
curl -X POST https://rl-server.kindlyscientific.com/run -F "file=@/home/val/Downloads/test.json"
>> {"status":"started","timestamp":"2026_05_22_15_05_17"}
curl "https://rl-server.kindlyscientific.com/check?timestamp=2026_05_22_15_05_17"
>> {"timestamp":"2026_05_22_15_05_17","status":"running"}
>> {"timestamp":"2026_05_22_15_05_17","status":"completed"}
curl "https://rl-server.kindlyscientific.com/download?timestamp=2026_05_22_15_05_17" --output /home/val/Downloads/result.zip
```

Etc:
```
tmux
ctrl + b, d
tmux list-sessions
tmux attach -t 0
tmux attach -t 1
```