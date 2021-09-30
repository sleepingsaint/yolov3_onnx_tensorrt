# downloading the files using gdown 
# https://github.com/wkentaro/gdown

import os
import gdown
import shutil

# file ids
links = {
    "1P2pV7CTRZbBoqv3i3np2WwU9wCVyHIxG": ["yolov3TATA608_final.weights", "data/"],
    "1E5xeyAmZeL86JVMGK9maWerV4c3tvoWV": ["yolov3TATA608.cfg", "data/"],
    "154S6WyZqLeF3G3gPTMibGnLnVLcpmEwm": ["test_image.jpg", "data/"],
    "13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU": ["test_video.mp4", "data/"]
}

# download the files and save to respective folder
cwd = os.getcwd()
for id in links:
  filename = links[id][0]
  filepath = links[id][1]
  destination = os.path.join(cwd, filepath)

  if not os.path.isdir(destination):
    os.mkdir(destination)

  url = f"https://drive.google.com/uc?id={id}"
  gdown.download(url, filename, quiet=False)
  shutil.move(os.path.join(cwd, filename), destination)