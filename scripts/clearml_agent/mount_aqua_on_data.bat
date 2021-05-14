rem Download rclone from: https://rclone.org/
rem Extract into a folder.
rem Copy this file and rclone.conf to the new folder
rem You will also need the api key and secret installed through awscli for this to work.
rem Note that you can map \\wsl$\Ubuntu to a drive letter and then browse the data.
call wsl.exe fusermount -uz /data/aqua-ml-data
call wsl.exe rclone mount --allow-other s3aqua:aqua-ml-data /data/aqua-ml-data

pause