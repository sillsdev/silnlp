# B2/MinIO bucket setup

We use Backblaze B2 and MinIO storage for storing our experiment data. Here is some workspace setup to enable a decent workflow.

### Note For MinIO setup

In order to access the MinIO bucket locally, you must have a VPN connected to its network. If you need VPN access, please reach out to an SILNLP dev team member.

### Install and configure rclone

**Windows**

The following will mount /silnlp on your B drive or /nlp-research on your M drive and allow you to explore, read and write.
* Install WinFsp: http://www.secfs.net/winfsp/rel/  (Click the button to "Download WinFsp Installer" not the "SSHFS-Win (x64)" installer)
* Download rclone from: https://rclone.org/downloads/
* Unzip to your desktop (or some convient location). 
* Add the folder that contains rclone.exe to your PATH environment variable.
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~\AppData\Roaming\rclone` (creating folders if necessary)
* Add your credentials in the appropriate fields in `~\AppData\Roaming\rclone`
* Take the `scripts/rclone/mount_b2_to_b.bat` and `scripts/rclone/mount_minio_to_m.bat` file from this SILNLP repo and copy it to the folder that contains the unzipped rclone.
* Double-click either bat file. A command window should open and remain open. You should see something like, if running mount_b2_to_b.bat:
```
C:\Users\David\Software\rclone>call rclone mount --vfs-cache-mode full --use-server-modtime b2silnlp:silnlp B: 
The service rclone has been started.
```

**Linux / macOS**

The following will mount /silnlp to a B folder or /nlp-research to a M folder in your home directory and allow you to explore, read and write.
* For macOS, first download and install macFUSE: https://osxfuse.github.io/
* Download rclone from: https://rclone.org/install/
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~/.config/rclone/rclone.conf` (creating folders if necessary)
* Add your credentials in the appropriate fields in `~/.config/rclone/rclone.conf`
* Create a folder called "B" or "M" in your user directory 
* Run the following command for B2:
   ```
   rclone mount --vfs-cache-mode full --use-server-modtime b2silnlp:silnlp ~/B
   ```
* OR run the following command for MinIO:
   ```
   rclone mount --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/M
   ```
### To start B: and/or M: drive on start up

**Windows**

Put a shortcut to the mount_b2_to_b.bat and/or mount_minio_to_m.bat file in the Startup folder.
* In Windows Explorer put `shell:startup` in the address bar or open `C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
* Right click to add a new shortcut. Choose `mount_b2_to_b.bat` and/or `mount_minio_to_m.bat` as the target, you can leave the name as the default.  

Now your B2 and/or MinIO bucket should be mounted as B: or M: drive, respectively, when you start Windows.

**Linux / macOS**
* Run `crontab -e`
* For B2, paste `@reboot rclone mount --vfs-cache-mode full --use-server-modtime b2silnlp:silnlp ~/B` into the file, save and exit
* For MinIO, paste `@reboot rclone mount --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/M` into the file, save and exit
* Reboot Linux / macOS

Now your B2 and/or MinIO bucket should be mounted as ~/B or ~/M respectively when you start Linux / macOS.