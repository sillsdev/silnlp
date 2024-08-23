rem Install rclone
rem   get rclone from https://rclone.org/downloads/
rem   extract the files to a folder
rem   then move this bat file to the folder where you  run this bat file to start the service
rem   --use-server-modtime flag speeds up displaying large numbers of files.  Not exactly mod time, but close enough.

rem configure rclone
rem   copy the adjacent file "rclone.conf" to: C:\Users\<username>\AppData\Roaming\rclone\rclone.conf
rem   copy your key and secret to rclone.conf

rem run rclone - execute this file in the rclone folder

call rclone mount --vfs-cache-mode full --use-server-modtime s3silnlp:silnlp S: 