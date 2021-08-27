import os
from braceexpand import braceexpand

try:
    import boto3
except:
    print("boto3 not installed")
    pass


def aws_download_to_local(
    remote_filename: str,
    local_filename: str,
    s3_resource: boto3.resource = None,
    bucket: str = "solar-pv-nowcasting-data",
):
    """
    Download file from aws
    @param remote_filename: the aws file name, should start with s3://
    @param local_filename:
    @param s3_resource: s3 resource, means a new one doesnt have to be made everytime.
    @param bucket: The s3 bucket name, from which to load the file from.
    """

    if s3_resource is None:
        s3_resource = boto3.resource("s3")

    # see if file exists in s3
    s3_resource.Object(bucket, remote_filename).load()

    # download file
    s3_resource.Bucket(bucket).download_file(remote_filename, local_filename)
