# Instructions for setting up Clear-ML on Windows.

These were written and tested for use with Windows 10.
See [Clear-ML Linux setup](clear_ml_linux_setup.md) for instructions to set up Clear-ML on linux.

## Install the clearml python package.
Open a command window and use pip to install Clear-ML.  
`pip install clearml`

## Add your AWS storage vault credentials (If using AWS S3).
1. Login to [Clear-ML](https://app.sil.hosted.allegro.ai) login with your work email address.
2. Go to the [workspace settings](https://app.sil.hosted.allegro.ai/settings/workspace-configuration).
3. At the top of the page you should see the configuration vault. If you don't see the configuration vault it is probably the case that you are not logged in to the Enterprise version of Clear-ML.  
   Add your aws key and secret and the region to the configuration vault using this format:
```
sdk {
  aws {
    s3 {
      key: "xxxxxxxxxxxxxxxxxx"
      secret: "xxxxxxxxxxxxxxxxxx"
      region: "us-east-1"
    }
  }
}
```

## Create your Clear-ML credentials
1. In a command window enter `clearml-init` you should be prompted to `Paste copied configuration here:`
2. On the [workspace settings](https://app.sil.hosted.allegro.ai/settings/workspace-configuration) webpage click `Create new credentials`.

They'll look something like this:
```
api { 
  web_server: https://app.sil.hosted.allegro.ai
  api_server: https://api.sil.hosted.allegro.ai
  files_server: https://files.sil.hosted.allegro.ai
  credentials {
    "access_key" = "xxxxxxxxxxxxxxxxxx"
    "secret_key"  = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }
}
```
3. Use the button to copy the new credentials to your clipboard. 
4. Paste them into the command window.
5. This will create a clearml.conf file in your home directory i.e.  C:\Users\<Username>\clearml.conf
6. If this file already exists the `clearml-init` command will invite you to edit it.  You may find it easier to delete it and run through these instructions, or you can put the copied details into the existing file in the required format.
