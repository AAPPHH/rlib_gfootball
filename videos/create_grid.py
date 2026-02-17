import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_DIR = r"C:\clones\rlib_gfootball\videos\videos"
OUTPUT_FILE = r"C:\clones\rlib_gfootball\videos\grid_output_max.mp4"
EXTENSION = "avi"

CELL_W, CELL_H = 1280, 720
FINAL_W, FINAL_H = 3840, 2160
PARALLEL = 3

# Maximum Quality NVENC Settings
ENC_OPTS = [
    "-c:v", "h264_nvenc",
    "-preset", "p7",           # höchste Qualität
    "-tune", "hq",             # high quality tuning
    "-rc", "vbr",              # variable bitrate
    "-cq", "15",               # constant quality (niedriger = besser, 15-18 ist sehr gut)
    "-b:v", "0",               # keine Bitrate-Limite
    "-maxrate", "100M",        # max burst
    "-bufsize", "200M",        # buffer
    "-profile:v", "high",
    "-level", "5.2",
    "-rc-lookahead", "32",     # bessere Szenen-Erkennung
    "-spatial-aq", "1",        # adaptive quantization
    "-temporal-aq", "1",       # temporal AQ
    "-aq-strength", "8",       # AQ stärke
    "-b_ref_mode", "middle",   # bessere B-Frame Referenzen
    "-bf", "4",                # mehr B-Frames
]


def get_duration(filepath):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0


def get_fps(filepath):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)
    ], capture_output=True, text=True)
    return result.stdout.strip() or "30"


def run_ffmpeg(args):
    return subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args).returncode == 0


def render_slot(slot_idx, slot_videos, temp_dir):
    concat_file = temp_dir / f"slot_{slot_idx}.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for video in slot_videos:
            escaped = str(video).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    
    slot_output = temp_dir / f"slot_{slot_idx}.mp4"
    
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-vf", f"scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
               f"pad={CELL_W}:{CELL_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        *ENC_OPTS, "-an", str(slot_output)
    ])
    return slot_idx, slot_output


def pad_slot(slot_idx, slot_file, pad_dur, temp_dir):
    fps = get_fps(slot_file)
    padded_file = temp_dir / f"slot_{slot_idx}_padded.mp4"
    
    run_ffmpeg([
        "-i", str(slot_file),
        "-f", "lavfi", "-t", str(pad_dur),
        "-i", f"color=black:size={CELL_W}x{CELL_H}:rate={fps}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]",
        "-map", "[out]", *ENC_OPTS, str(padded_file)
    ])
    return slot_idx, padded_file


def main():
    input_dir = Path(INPUT_DIR)
    output_file = Path(OUTPUT_FILE)
    
    files = sorted(input_dir.glob(f"*.{EXTENSION}"))
    
    if not files:
        print(f"Keine .{EXTENSION} Dateien gefunden")
        return
    
    print(f"Input: {len(files)} Dateien | MAX QUALITY | 4K")
    temp_dir = Path(tempfile.mkdtemp(prefix="grid_"))
    
    try:
        slots = [[] for _ in range(9)]
        for i, f in enumerate(files):
            slots[i % 9].append(f)
        
        slot_files = [None] * 9
        print("Slots: ", end="", flush=True)
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futures = {ex.submit(render_slot, i, slots[i], temp_dir): i for i in range(9)}
            for future in as_completed(futures):
                idx, output = future.result()
                slot_files[idx] = output
                print(f"{idx}✓ ", end="", flush=True)
        print()
        
        durations = [get_duration(f) for f in slot_files]
        max_duration = max(durations)
        
        pad_tasks = [(i, slot_files[i], max_duration - dur) 
                     for i, dur in enumerate(durations) if dur < max_duration - 0.1]
        
        if pad_tasks:
            print("Padding: ", end="", flush=True)
            with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
                futures = {ex.submit(pad_slot, i, f, pad, temp_dir): i 
                          for i, f, pad in pad_tasks}
                for future in as_completed(futures):
                    idx, padded = future.result()
                    slot_files[idx] = padded
                    print(f"{idx}✓ ", end="", flush=True)
            print()
        
        layout = (f"0_0|{CELL_W}_0|{CELL_W*2}_0|"
                  f"0_{CELL_H}|{CELL_W}_{CELL_H}|{CELL_W*2}_{CELL_H}|"
                  f"0_{CELL_H*2}|{CELL_W}_{CELL_H*2}|{CELL_W*2}_{CELL_H*2}")
        
        inputs = []
        for f in slot_files:
            inputs.extend(["-i", str(f)])
        
        print("Final grid...", end=" ", flush=True)
        run_ffmpeg([
            *inputs,
            "-filter_complex",
            f"[0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v][8:v]xstack=inputs=9:layout={layout}[grid];"
            f"[grid]scale={FINAL_W}:{FINAL_H}:flags=lanczos[out]",
            "-map", "[out]", *ENC_OPTS, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_file)
        ])
        print("✓")
        
        size_gb = output_file.stat().st_size / 1024**3
        print(f"Output: {output_file} ({size_gb:.2f} GB)")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()