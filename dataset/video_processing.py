import os
import subprocess
from pathlib import Path

def process_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extensions = ('.mp4', '.avi', '.mov', '.mkv')

    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() in extensions:
            input_file = str(file_path)
            output_file = os.path.join(output_dir, f"drop_frames_30s_{file_path.name}")
            
            print(f"Processing: {file_path.name}...")

            # FFmpeg Command Explanation:
            # -ss 0 -t 60: Take the first 60 seconds of the input.
            # select='not(mod(n,2))': Selects frame 0, 2, 4, 6... (effectively dropping every other frame).
            # setpts=0.5*PTS: Adjusts the timestamps so the video plays at the correct speed for the remaining frames.
            
            command = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-ss', '00:00:00',
                '-t', '15', 
                '-filter:v', "select='not(mod(n,2))',setpts=0.5*PTS",
                '-an', # Removing audio entirely since it's not needed for depth maps
                '-crf', '23',
                '-preset', 'veryfast',
                output_file
            ]

            try:
                # We use capture_output=True to catch errors, but it won't crash on 'no audio' now
                subprocess.run(command, check=True, capture_output=True)
                print(f"Success! Final 30s video (dropping frames) saved to: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {file_path.name}: {e.stderr.decode()}")

                
if __name__ == "__main__":
    # Ensure these folders exist in your directory
    INPUT_DIRECTORY = "/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/videos" 
    OUTPUT_DIRECTORY = "./processed_videos"
    
    process_videos(INPUT_DIRECTORY, OUTPUT_DIRECTORY)