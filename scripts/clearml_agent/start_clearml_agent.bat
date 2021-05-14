rem call  "mount_aqua_on_data.bat" before this file is called.
rem move clearml.conf to C:/Users/Username/clearml.conf
call clearml-agent daemon --cpu-only --queue cpu_only --docker
pause