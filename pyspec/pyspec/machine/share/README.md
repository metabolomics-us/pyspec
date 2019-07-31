# Share

this module contains a couple of useful utilities to share datasets with third parties. By utilizing a central storage
interface to upload and receive data.

Custom implementations are provided for amazon S3.

# S3

This provides you with read write and optional read only access to a global S3 bucket. This bucket should be setup with
permissions like the following

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AddPerm",
            "Effect": "Allow",
            "Principal": "*",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectVersion"
            ],
            "Resource": "arn:aws:s3:::wcmc-machine-datasets/*",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "128.120.143.0/24"
                }
            }
        },
        {
            "Sid": "AddPerm",
            "Effect": "Allow",
            "Principal": "*",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::wcmc-machine-datasets",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "128.120.143.0/24"
                }
            }
        }
    ]
}
```

To make it publically accessible, but limit it to a certain ip range.