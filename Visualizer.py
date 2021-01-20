import subprocess, os
thisdir = os.getcwd()
subprocess.run(f'CD {thisdir}', shell=True)
subprocess.run('activate tensorflow', shell=True)
subprocess.run('tensorboard --logdir tb_callback_dir',shell=True)


## Change directory to your current working directory
## activate your venv
## type in: tensorboard --logdir tb_callback_dir
## a websocket will be hosted on this IP: http://localhost:6006