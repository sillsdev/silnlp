rem move clearml.conf to C:/Users/Username/clearml.conf
rem
rem ClearML credentials
rem Create new clearml credentials in https://app.sil.hosted.allegro.ai/profile
rem in clearml.conf, replace the credentials with the access and private keys that you created. 
rem 
rem SSH key management
rem make your ssh credentials by opening up git bash and typing: ssh-keygen -t rsa
rem The standard git keys that you just created will be shared with the docker image.  The permissions will be set properly with extra_docker_shell_script.
rem
rem Google Drive mounting
rem you can test the mounting in docker by executing the following:
rem docker run -it -v "G:\Shared drives\AQUA":/data/aqua-ml-data silintlai/machine-silnlp:master-latest ls /data/aqua-ml-data
rem see if the folders in aqua S3 bucket are displayed.
call clearml-agent daemon --cpu-only --queue cpu_only --docker
pause