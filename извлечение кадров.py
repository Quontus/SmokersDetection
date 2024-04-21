from moviepy.editor import VideoFileClip
import numpy as np
import os
from datetime import timedelta


def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    
    ms = round(int(ms) / 10000)
    return f"{result}.{ms:02}".replace(":", "-")

def main(video_file):
    video_clip = VideoFileClip(video_file)
    filename, _ = os.path.splitext(video_file)

    if not os.path.isdir(filename):
        os.mkdir(filename)

    SAVING_FRAMES_PER_SECOND = min(video_clip.fps, 3) #минимальное количество секунд
    step = video_clip.fps if SAVING_FRAMES_PER_SECOND == 0 else SAVING_FRAMES_PER_SECOND

    for current_duration in np.arange(0, video_clip.duration, step):
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(filename, f"foto__{frame_duration_formatted}.jpg") #минуты-секунды

        video_clip.save_frame(frame_filename, current_duration)


video_file = 'Девушка курит на улице (360p).mp4'
main(video_file)
