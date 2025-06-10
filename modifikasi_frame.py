#!/usr/bin/env python3

import os
import subprocess
import json
import sys
from pathlib import Path

def get_video_frame_count(video_path):
    """Get the total number of frames in a video file using ffprobe."""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            nb_frames = data['streams'][0].get('nb_frames')
            if nb_frames:
                return int(nb_frames)
        
        # Fallback method: count frames manually (slower but more reliable)
        print("  📊 Using manual frame counting (this may take a moment)...")
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-csv=p=0',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        frame_count = result.stdout.strip()
        
        if frame_count and frame_count.isdigit():
            return int(frame_count)
        else:
            return None
            
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error getting frame count for {video_path}: {e}")
        return None

def is_convertible_frame_count(frame_count):
    """Check if frame count is 24, 25, or 26."""
    if frame_count is None:
        return False
    
    return frame_count in [24, 25, 26]

def convert_to_25_frames(input_path, output_path):
    """Convert video to exactly 25 frames using ffmpeg."""
    try:
        # Method 1: Use select filter to get exactly 25 frames
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-vf', 'select=lt(n\\,25)',
            '-c:a', 'copy',
            str(output_path),
            '-y',  # Overwrite output file
            '-loglevel', 'error'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        if e.stderr:
            print(f"FFmpeg error: {e.stderr}")
        return False

def delete_video_and_txt(video_path, txt_path):
    """Delete both video and corresponding txt file."""
    try:
        if video_path.exists():
            video_path.unlink()
            print(f"  ✓ Deleted {video_path}")
        
        if txt_path.exists():
            txt_path.unlink()
            print(f"  ✓ Deleted {txt_path}")
        
        return True
        
    except OSError as e:
        print(f"Error deleting files: {e}")
        return False

def check_dependencies():
    """Check if required tools are available."""
    dependencies = ['ffmpeg', 'ffprobe']
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(dep)
    
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        print("Please install ffmpeg: brew install ffmpeg")
        return False
    
    return True

def process_videos(base_directory, video_pattern="*.mp4"):
    """Process all video files matching the pattern in subdirectories."""
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_directory}' does not exist")
        return False
    
    # Debug: Show directory structure
    print(f"Scanning directory: {base_path}")
    print(f"Looking for video files matching: {video_pattern}")
    print("Directory structure:")
    
    video_files = []
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        level = len(root_path.parts) - len(base_path.parts)
        indent = "  " * level
        print(f"{indent}{root_path.name}/")
        
        # Look for video files matching the pattern
        for file in files:
            file_path = root_path / file
            # Check if it's a video file (mp4, mov, avi, etc.)
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv')):
                video_files.append(file_path)
                print(f"{indent}  📹 Found: {file}")
                
                # Look for corresponding txt file
                txt_name = file_path.stem + '.txt'  # Same name but .txt extension
                txt_path = root_path / txt_name
                if txt_path.exists():
                    print(f"{indent}  📄 Found corresponding: {txt_name}")
                else:
                    print(f"{indent}  ⚠️  No corresponding txt file: {txt_name}")
            else:
                # Show other files for debugging
                sub_indent = "  " * (level + 1)
                print(f"{sub_indent}{file}")
    
    print(f"\nTotal video files found: {len(video_files)}")
    
    if not video_files:
        print("No video files found in any subdirectories")
        return True
    
    print(f"Found {len(video_files)} video files to process")
    print("=" * 50)
    
    processed_count = 0
    for video_path in video_files:
        # Find corresponding txt file (same name but .txt extension)
        txt_path = video_path.parent / (video_path.stem + '.txt')
        
        print(f"Processing: {video_path.relative_to(base_path)}")
        
        # Check if corresponding txt file exists
        if not txt_path.exists():
            print(f"  ⚠️  Warning: {txt_path} not found, skipping...")
            continue
        
        # Get frame count
        frame_count = get_video_frame_count(video_path)
        
        if frame_count is None:
            print("  ❌ Could not determine frame count, skipping...")
            continue
        
        print(f"  📊 Total frames: {frame_count}")
        
        # Check if frame count is convertible (24, 25, or 26)
        if is_convertible_frame_count(frame_count):
            
            if frame_count == 25:
                print("  ✅ Already has 25 frames, no conversion needed")
            else:
                print(f"  🔄 Converting from {frame_count} frames to 25 frames...")
                
                # Create temporary file for conversion
                temp_path = video_path.parent / f"{video_path.stem}_temp{video_path.suffix}"
                
                if convert_to_25_frames(video_path, temp_path):
                    # Replace original with converted video
                    try:
                        video_path.unlink()  # Delete original
                        temp_path.rename(video_path)  # Rename temp to original
                        print("  ✅ Successfully converted to 25 frames")
                    except OSError as e:
                        print(f"  ❌ Error replacing original file: {e}")
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                else:
                    print("  ❌ Error converting video")
                    # Clean up temp file if it exists
                    if temp_path.exists():
                        temp_path.unlink()
        else:
            print(f"  ❌ Frame count ({frame_count}) not 24/25/26 frames, deleting video and txt...")
            delete_video_and_txt(video_path, txt_path)
        
        processed_count += 1
        print()  # Empty line for readability
    
    print(f"Processed {processed_count} video files")
    return True

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python process_videos.py <directory_path> [video_pattern]")
        print("Examples:")
        print("  python process_videos.py /path/to/folder                    # Find all video files")
        print("  python process_videos.py /path/to/folder '*.mp4'           # Find only .mp4 files")
        print("  python process_videos.py /path/to/folder 'video_*.mp4'     # Find video_*.mp4 files")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    directory = sys.argv[1]
    video_pattern = sys.argv[2] if len(sys.argv) > 2 else "*.mp4"
    
    print(f"Starting video processing in: {directory}")
    print("=" * 50)
    
    success = process_videos(directory, video_pattern)
    
    print("=" * 50)
    if success:
        print("✅ Video processing completed!")
    else:
        print("❌ Video processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()


