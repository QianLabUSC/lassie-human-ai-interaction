import os
import shutil
import json
import cv2
from constants import USER_STUDY_LOG


class Logger:
    def __init__(self, config, filename, agent1=None, agent2=None, video_record=False):
        self.participant_id = config.participant_id
        self.json_filename = filename+'.json'
        self.filename = filename
        self.video_record = config.record_video
        # create log folder
        self.log_folder = os.path.join(
            USER_STUDY_LOG, str(self.participant_id))
        print(self.log_folder)
        self.img_dir = os.path.join(self.log_folder, 'img')

        if not os.path.exists(self.log_folder):
            # shutil.rmtree(self.log_folder)
            os.makedirs(self.log_folder)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        self.img_name = lambda timestep: f"{self.img_dir}/{int(timestep*10):05d}.png"

        # game info
        # self.env = config.base_env
        self.layout_name = config.layout_name
        self.agent1 = agent1
        self.agent2 = agent2
        self.episode = []
        
    def save_log_as_pickle(self):
        with open(os.path.join(self.log_folder, self.json_filename), 'w') as file:
            json.dump({"layout_name": self.layout_name,
                       "participant_id": self.participant_id,
                       "total_time": self.env.state.timestep,
                      "episode": self.episode}, file)
        print(f"Pickle log saved to {self.json_filename}")

    """
        Create video from images 
    """
    def create_video(self):
        images = [img for img in os.listdir(self.img_dir) if img.endswith(".png")]
        images.sort()  # Ensure images are sorted in the correct order
        if len(images) == 0:
            print("No images found to create video.")
            return
        frame = cv2.imread(os.path.join(self.img_dir, images[0]))
        height, width, layers = frame.shape
        video_name = '{}{}.mp4'.format(self.log_folder + '/', self.filename)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 container
        video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.img_dir, image)))

        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(self.img_dir)