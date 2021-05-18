rem call  "mount_aqua_on_data.bat" before this file is called.
rem move clearml.conf to C:/Users/Username/clearml.conf
rem you can test the mounting in docker by executing the following:
rem docker run -it -v \\wsl$\Ubuntu\data:/data silintlai/machine-silnlp:master-latest ls /data/aqua-ml-data
rem see if the folders in aqua S3 bucket are displayed.
call clearml-agent daemon --cpu-only --queue cpu_only --docker
pause