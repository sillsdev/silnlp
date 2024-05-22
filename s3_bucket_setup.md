# S3 bucket setup

We use Amazon S3 storage for storing our experiment data. Here is some workspace setup to enable a decent workflow.

### Install and configure AWS S3 storage
* Install the aws-cli from: https://aws.amazon.com/cli/
* In cmd, type: `aws configure` and enter your AWS access_key_id and secret_access_key and the region (we use region = us-east-1).
* The aws configure command will create a folder in your home directory named '.aws' it should contain two plain text files named 'config' and 'credentials'. The config file should contain the region and the credentials file should contain your access_key_id and your secret_access_key.
(Home directory on windows is usually C:\Users\<Username>\ and on linux it is /home/username)

### Install and configure rclone

**Windows**

The following will mount /aqua-ml-data on your S drive and allow you to explore, read and write.
* Install WinFsp: http://www.secfs.net/winfsp/rel/  (Click the button to "Download WinFsp Installer" not the "SSHFS-Win (x64)" installer)
* Download rclone from: https://rclone.org/downloads/
* Unzip to your desktop (or some convient location). 
* Add the folder that contains rclone.exe to your PATH environment variable.
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~\AppData\Roaming\rclone` (creating folders if necessary)
* Add your credentials in the appropriate fields in `~\AppData\Roaming\rclone`
* Take the `scripts/rclone/mount_to_s.bat` file from this SILNLP repo and copy it to the folder that contains the unzipped rclone.
* Double-click the bat file. A command window should open and remain open. You should see something like:
```
C:\Users\David\Software\rclone>call rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data S:
The service rclone has been started.
```

**Linux**

The following will mount /aqua-ml-data to an S folder in your home directory and allow you to explore, read and write.
* Download rclone from: https://rclone.org/install/
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~/.config/rclone/rclone.conf` (creating folders if necessary)
* Add your credentials in the appropriate fields in `~/.config/rclone/rclone.conf`
* Create a folder called "S" in your user directory 
* Run the following command:
   ```
   rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data ~/S
   ```
### To start S: drive on start up

**Windows**

Put a shortcut to the mount_to_s.bat file in the Startup folder.
* In Windows Explorer put `shell:startup` in the address bar or open `C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
* Right click to add a new shortcut. Choose `mount_to_s.bat` as the target, you can leave the name as the default.  

Now your AWS S3 bucket should be mounted as S: drive when you start Windows.

**Linux**
* Run `crontab -e`
* Paste `@reboot rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data ~/S` into the file, save and exit
* Reboot Linux

Now your AWS S3 bucket should be mounted as ~/S when you start Linux.