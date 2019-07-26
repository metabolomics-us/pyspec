import boto3

from pyspec import config
from pyspec.machine.share import Share
import shutil
import os
import zipfile


class S3Share(Share):
    """
    shares datasets to S3 on AWS
    """

    def zipdir(self, path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                # we do not want to zip the trained models
                if not file.endswith("model.h5"):
                    ziph.write(os.path.join(root, file))

    def exists(self, name: str) -> bool:
        """
        checks if a given directory exists on the remote server
        :param name:
        :return:
        """
        content = self.client.head_object(Bucket=self.bucket_name, Key="{}/training_data.zip".format(name))
        if content.get('ResponseMetadata', None) is not None:
            return True
        else:
            return False

    def retrieve(self, name: str, root_folder: str = 'datasets'):
        # 1. download folder

        # 2. unzip it

        pass

    def submit(self, name: str, root_folder: str = 'datasets'):
        # 1. zip folder with trained models

        path = "upload/{}".format(name)
        os.makedirs(path, exist_ok=True)

        with zipfile.ZipFile('upload/{}/training_data.zip'.format(name), 'w', zipfile.ZIP_DEFLATED) as file:
            self.zipdir("{}/{}".format(root_folder, name), file)

        # 2. copy model into our upload folder
        for root, dirs, files in os.walk("{}/{}".format(root_folder, name)):
            for file in files:
                # we do not want to zip the trained models
                if file.endswith("model.h5"):
                    shutil.copyfile(src=os.path.join(root, file), dst=os.path.join(path, file))
        # 3. upload it
        for subdir, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(subdir, file)
                self.client.upload_file(full_path, self.bucket_name, full_path.replace("upload/", ""))

        pass

    def __init__(self, config_file: str = "shares.ini"):
        """
        
        :param config_file: 
        """

        bucket_config = config.config(config_file, "s3-bucket")
        self.bucket_name = bucket_config["name"]
        self.constraint = bucket_config["constraint"]
        self.s3 = boto3.resource('s3')
        self.client = boto3.client('s3')

        try:
            self.client.create_bucket(Bucket=self.bucket_name, CreateBucketConfiguration={
                'LocationConstraint': self.constraint})
        except Exception as e:
            print("sorry this bucket caused an error - this mean it exist, no reason to worry")
