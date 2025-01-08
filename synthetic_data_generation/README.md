# Synthetic Data Generation

1) Install requirements. Python 3.10 is necessary
`pip install -r requirements.txt`

# Fitting the avatars
## Using RANDOM PEOPLE dataset
1) clone Exavatar using 'git clone https://github.com/mks0601/ExAvatar_RELEASE.git'
2) Go to the folder 'cd ./ExAvatar_RELEASE/fitting/data/Custom/data' and download RANDOM people dataset using 'xx'.

## Using your own avatars
If you would like to to add more monocular videos, place them in './ExAvatar_RELEASE/fitting/data/Custom/data' in the same data structure
3) remove old 'run.py' using 'rm ./ExAvatar_RELEASE/fitting/tools/run.py'
4) move new 'run.py' in 'tools' folder using 'mv Pose-Transfer-synthetic-data/synthetic_data_generation/run.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools'
5) Go to 'cd ./ExAvatar_RELEASE/fitting/tools'
6) Run 'python run.py --rootpath path/to/your/video_folder'

# Fitting the motion videos
1) place your dataset folder with '.mp4' videos in 'cd ./ExAvatar_RELEASE/fitting/data/Custom/data/'
2) To create usable dataset for Exavatar change paths in 'mp4.sh' and run 'bash mp4.sh'
3) move 'motion_run.py' in 'tools' folder using 'mv Pose-Transfer-synthetic-data/synthetic_data_generation/motion_ run.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools'
4) Go to 'cd ./ExAvatar_RELEASE/fitting/tools'
5) Run 'motion_run.py --rootpath path/to/your/video_folder'
   

