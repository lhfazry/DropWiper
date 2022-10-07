#!/bin/bash

# login
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=lhfazry&password=Z@2Thi3b&submit=Login' https://www.cityscapes-dataset.com/login/

# gtFine_trainvaltest.zip
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

# leftImg8bit_trainvaltest.zip
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

# camera_trainvaltest.zip
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=8

# extract
unzip gtFine_trainvaltest.zip -d datasets
unzip leftImg8bit_trainvaltest.zip -d datasets
unzip camera_trainvaltest.zip -d datasets

exit 1