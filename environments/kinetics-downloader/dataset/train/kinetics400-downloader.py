import pandas as pd
import os
import subprocess
import time
import random

# === CONFIG ===
CSV_PATH = "/home/richtsai1103/CRL/minco/environments/kinetics-downloader/dataset/train/train.csv"
LABEL = "driving car"
OUT_DIR = "/storage/ssd1/richtsai1103/kinetics400"
NUM_VIDEOS = 100  # Change this to download more
MAX_RETRIES = 3
DELAY_RANGE = (1, 3)  # Random delay between downloads

# === SETUP ===
os.makedirs(OUT_DIR, exist_ok=True)

def download_video_with_retry(url, temp_filename, max_retries=MAX_RETRIES):
    """Download video with retry logic and better error handling"""
    for attempt in range(max_retries):
        try:
            # Try different format selectors in order of preference
            format_selectors = [
                "best[height<=480][ext=mp4]",  # Best MP4 up to 480p
                "best[ext=mp4]",               # Best MP4 any resolution
                "best[height<=720]",           # Best format up to 720p
                "best"                         # Any best format
            ]
            
            for fmt in format_selectors:
                try:
                    subprocess.run(
                        ["yt-dlp", "-f", fmt, url, "-o", temp_filename],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=60  # 60 second timeout
                    )
                    return True
                except subprocess.CalledProcessError:
                    continue
            
            # If all formats failed, try one more time with default settings
            subprocess.run(
                ["yt-dlp", url, "-o", temp_filename],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60
            )
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if attempt < max_retries - 1:
                wait_time = random.uniform(2, 5)
                print(f"    Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return False
    
    return False

def process_video(video_id, start, end, temp_filename, output_filename):
    """Process video with ffmpeg with better error handling"""
    duration = end - start
    
    try:
        # Check if temp file exists and has content
        if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
            print(f"    ❌ Temp file invalid or empty")
            return False
        
        # Run ffmpeg with timeout
        subprocess.run(
            [
                "ffmpeg", "-ss", str(start), "-i", temp_filename,
                "-t", str(duration), "-vf", "scale=84:84",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy", "-y", output_filename  # -y to overwrite
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30  # 30 second timeout for ffmpeg
        )
        
        # Verify output file was created and has content
        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            return True
        else:
            print(f"    ❌ Output file invalid or empty")
            return False
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    ❌ FFmpeg processing failed: {e}")
        return False

# Read CSV
try:
    df = pd.read_csv(CSV_PATH, sep=',')
    print(f"Loaded {len(df)} total videos")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Filter and process
df_filtered = df[df["label"] == LABEL].head(NUM_VIDEOS)
print(f"Found {len(df_filtered)} videos for label '{LABEL}'")

successful_downloads = 0
failed_downloads = 0

for idx, row in df_filtered.iterrows():
    video_id = row["youtube_id"]
    start = int(row["time_start"])
    end = int(row["time_end"])
    temp_filename = f"temp_{video_id}.mp4"
    output_filename = os.path.join(OUT_DIR, f"{video_id}_{start}_{end}.mp4")

    print(f"\n[{idx}] Processing video {video_id}")
    
    # Skip if already exists
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        print(f"    ✅ Already exists: {output_filename}")
        successful_downloads += 1
        continue

    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        # Download with retry
        print(f"    Downloading {url}")
        if download_video_with_retry(url, temp_filename):
            print(f"    Download successful, processing...")
            
            # Process video
            if process_video(video_id, start, end, temp_filename, output_filename):
                print(f"    ✅ Saved to: {output_filename}")
                successful_downloads += 1
            else:
                print(f"    ❌ Processing failed for: {video_id}")
                failed_downloads += 1
        else:
            print(f"    ❌ Download failed for: {video_id}")
            failed_downloads += 1

    except Exception as e:
        print(f"    ❌ Unexpected error for {video_id}: {e}")
        failed_downloads += 1

    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
    
    # Add delay between downloads to be respectful
    if idx < len(df_filtered) - 1:  # Don't delay after last video
        delay = random.uniform(*DELAY_RANGE)
        time.sleep(delay)

print(f"\n=== SUMMARY ===")
print(f"Successful: {successful_downloads}")
print(f"Failed: {failed_downloads}")
print(f"Total processed: {successful_downloads + failed_downloads}")
print(f"Success rate: {successful_downloads/(successful_downloads + failed_downloads)*100:.1f}%")