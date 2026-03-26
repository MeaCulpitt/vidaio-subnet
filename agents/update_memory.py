#!/usr/bin/env python3
"""
SN85 Miner Memory Updater
Runs every 30 minutes via pm2 cron (xx:15 and xx:45).
Updates /root/MEMORY.md with current service health, GPU state, and last task info.
"""
import subprocess
import re
import json
from datetime import datetime, timezone
from pathlib import Path

MEMORY_PATH = Path("/root/MEMORY.md")
PM2_LOG_DIR = Path("/root/.pm2/logs")
SERVICES = ["redis", "sn85-miner", "video-compressor", "video-deleter", "video-upscaler"]


def run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def get_pm2_status():
    """Return dict of service_name -> {status, uptime, restarts, cpu, mem}."""
    raw = run("pm2 jlist")
    try:
        procs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {s: {"status": "unknown"} for s in SERVICES}

    result = {}
    for p in procs:
        name = p.get("name", "")
        env = p.get("pm2_env", {})
        monit = p.get("monit", {})
        result[name] = {
            "status": env.get("status", "unknown"),
            "uptime": env.get("pm_uptime", 0),
            "restarts": env.get("restart_time", 0),
            "cpu": monit.get("cpu", 0),
            "mem_mb": round(monit.get("memory", 0) / 1024 / 1024, 1),
        }
    return result


def format_uptime(pm_uptime_ms):
    if not pm_uptime_ms:
        return "unknown"
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    diff_s = (now_ms - pm_uptime_ms) / 1000
    if diff_s < 60:
        return f"{int(diff_s)}s"
    elif diff_s < 3600:
        return f"{int(diff_s/60)}m"
    elif diff_s < 86400:
        return f"{diff_s/3600:.1f}h"
    else:
        return f"{diff_s/86400:.1f}d"


def get_gpu_state():
    """Return dict with temp, vram_used, vram_total, utilization."""
    raw = run("nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    if "ERROR" in raw or not raw:
        return {"temp": "?", "vram_used": "?", "vram_total": "?", "util": "?"}
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 4:
        return {"temp": "?", "vram_used": "?", "vram_total": "?", "util": "?"}
    return {
        "temp": f"{parts[0]}C",
        "vram_used": f"{parts[1]}MB",
        "vram_total": f"{parts[2]}MB",
        "util": f"{parts[3]}%",
    }


def get_last_task():
    """Parse upscaler logs for the most recent completed task."""
    log_path = PM2_LOG_DIR / "video-upscaler-error.log"
    if not log_path.exists():
        return None

    # Read last 200 lines
    raw = run(f"tail -200 {log_path}")
    if not raw:
        return None

    # Find last "Total e2e:" line
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| INFO .* Total e2e: ([\d.]+)s"
    matches = list(re.finditer(pattern, raw))
    if not matches:
        return None

    last = matches[-1]
    timestamp = last.group(1)
    duration = last.group(2)

    # Find the task type from nearby lines (look for x2/x4 path)
    line_start = raw.rfind("\n", 0, last.start()) + 1
    context_start = max(0, line_start - 500)
    context = raw[context_start:last.end()]

    task_type = "unknown"
    if "x4 path" in context or "x4 model" in context or "PLKSR x4" in context:
        task_type = "SD24K (x4)"
    elif "x2 path" in context or "nvidia-vfx x2" in context or "SPAN x2" in context:
        task_type = "SD2HD (x2)"

    return {"timestamp": timestamp, "duration": duration, "task_type": task_type}


def get_recent_crashes():
    """Find crash events since last memory update."""
    log_path = PM2_LOG_DIR / "video-upscaler-error.log"
    if not log_path.exists():
        return []

    raw = run(f"tail -500 {log_path}")
    if not raw:
        return []

    crashes = []
    # Look for ERROR lines (skip known noise)
    noise_patterns = ["UnknownSynapseError", "ValueError.*hotkey", "DeprecationWarning"]
    for line in raw.split("\n"):
        if "| ERROR |" not in line:
            continue
        if any(p in line for p in noise_patterns):
            continue
        # Extract timestamp and message
        m = re.match(r".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| ERROR .* - (.+)", line)
        if m:
            crashes.append({"timestamp": m.group(1), "message": m.group(2)[:120]})

    # Deduplicate by message, keep latest
    seen = {}
    for c in crashes:
        key = c["message"][:60]
        seen[key] = c
    return list(seen.values())[-5:]  # Keep last 5 unique


def read_existing_memory():
    if MEMORY_PATH.exists():
        return MEMORY_PATH.read_text()
    return ""


def update_memory():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Gather data
    pm2 = get_pm2_status()
    gpu = get_gpu_state()
    last_task = get_last_task()
    crashes = get_recent_crashes()

    existing = read_existing_memory()

    # Build the auto-updated section
    lines = []
    lines.append(f"## Auto-Status (updated {now})")
    lines.append("")

    # Service health
    lines.append("### Services")
    lines.append("| Service | Status | Uptime | Restarts | Mem |")
    lines.append("|---------|--------|--------|----------|-----|")
    all_ok = True
    for svc in SERVICES:
        info = pm2.get(svc, {"status": "missing"})
        status = info.get("status", "unknown")
        uptime = format_uptime(info.get("uptime", 0))
        restarts = info.get("restarts", "?")
        mem = f"{info.get('mem_mb', '?')}MB"
        emoji = "online" if status == "online" else f"**{status.upper()}**"
        if status != "online":
            all_ok = False
        lines.append(f"| {svc} | {emoji} | {uptime} | {restarts} | {mem} |")
    lines.append("")

    # GPU
    lines.append(f"### GPU: {gpu['temp']} | VRAM {gpu['vram_used']}/{gpu['vram_total']} | Util {gpu['util']}")
    lines.append("")

    # Last task
    if last_task:
        lines.append(f"### Last Task: {last_task['task_type']} at {last_task['timestamp']} ({last_task['duration']}s)")
    else:
        lines.append("### Last Task: none found in recent logs")
    lines.append("")

    # Recent crashes
    if crashes:
        lines.append("### Recent Errors")
        for c in crashes:
            lines.append(f"- `{c['timestamp']}` {c['message']}")
        lines.append("")
    else:
        lines.append("### Recent Errors: none")
        lines.append("")

    auto_section = "\n".join(lines)

    # Replace or append the auto-status section
    marker_start = "## Auto-Status"
    marker_end_candidates = [
        "## Subnet", "## PM2 Services", "## GPU", "## Upscaling Pipeline",
        "## What's Working", "## Key Changes", "## Benchmarks", "## Key Bottleneck",
        "## Key File Paths"
    ]

    if marker_start in existing:
        # Find the auto-status section and replace it
        start_idx = existing.index(marker_start)
        # Find the next section header after auto-status
        end_idx = len(existing)
        for marker in marker_end_candidates:
            pos = existing.find(marker, start_idx + len(marker_start))
            if pos != -1 and pos < end_idx:
                end_idx = pos
        existing = existing[:start_idx] + auto_section + "\n" + existing[end_idx:]
    else:
        # Insert after the first heading
        first_newline = existing.find("\n")
        if first_newline != -1:
            existing = existing[:first_newline+1] + "\n" + auto_section + "\n" + existing[first_newline+1:]
        else:
            existing = existing + "\n\n" + auto_section

    MEMORY_PATH.write_text(existing)
    print(f"[{now}] MEMORY.md updated — services: {'ALL OK' if all_ok else 'ISSUES DETECTED'}, "
          f"GPU: {gpu['temp']} {gpu['util']}")


if __name__ == "__main__":
    update_memory()
