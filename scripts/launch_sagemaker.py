"""
launch_sagemaker.py — AWS SageMaker Training Job launcher and live log tail

Starts AWS SageMaker jobs directly from your local machine with Python (boto3)
— no long AWS CLI commands or JSON files needed:
  1. Zero-Shot / Checkpoint evaluation (--mode eval)
  2. Fast image download and tar packaging (--mode download)
  3. CLIP LoRA fine-tuning training (--mode train)
It packages the code, uploads it to S3, and streams CloudWatch logs live to
your local terminal.

Usage examples:
  # 1. Using environment variables (the safe way for GitHub):
  $env:S3_BUCKET="your-s3-bucket-name"
  $env:SAGEMAKER_ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
  python scripts/launch_sagemaker.py --mode train --follow

  # 2. Or by passing the parameters directly:
  python scripts/launch_sagemaker.py --mode eval --bucket my-bucket --role-arn arn:aws:iam::123:role/MyRole --follow
"""

import argparse
from datetime import datetime
import io
import os
from pathlib import Path
import tarfile
import time

import boto3

# Official AWS PyTorch 2.1.0 + CUDA 12.1 Deep Learning Container image template
DLC_IMAGE_TEMPLATE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/"
    "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
)

# Project root (one level above the scripts/ directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project files shipped into the SageMaker container
SOURCE_FILES = [
    "train.py",
    "eval.py",
    "dataset.py",
    "download_images.py",
    "config.yaml",
]

# Minimal dependencies so the GPU-PyTorch build inside the container is untouched
CONTAINER_REQUIREMENTS = """\
transformers==4.41.2
peft==0.11.1
accelerate==0.30.1
pyarrow
pyyaml
aiohttp
pillow
tqdm
"""


def build_and_upload_source_archive(s3_client, bucket: str, s3_key: str) -> str:
    """
    Pack the local Python files and a container-safe requirements.txt into an
    in-memory .tar.gz archive and upload it straight to S3.
    Leaves no temporary files on the local disk.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for filename in SOURCE_FILES:
            file_path = PROJECT_ROOT / filename
            if file_path.exists():
                tar.add(str(file_path), arcname=filename)

        # Add container-compatible requirements.txt
        req_bytes = CONTAINER_REQUIREMENTS.encode("utf-8")
        info = tarfile.TarInfo(name="requirements.txt")
        info.size = len(req_bytes)
        tar.addfile(info, io.BytesIO(req_bytes))

    buffer.seek(0)
    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"[1/3] Source code packaged; uploading to S3 -> {s3_uri}")
    s3_client.upload_fileobj(buffer, bucket, s3_key)
    return s3_uri


def build_input_channels(bucket: str, mode: str) -> list[dict]:
    """
    Prepare the S3 data channels to mount, depending on the job type
    (eval / download / train).
    """
    channels = [
        {
            "ChannelName": "splits",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/data/processed/splits/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    ]

    if mode == "train":
        # For training, download all 81,054 images as a single archive in ~25 seconds:
        channels.append(
            {
                "ChannelName": "images_bundle",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/data/images_bundle.tar",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        )
    else:
        # In eval or download mode, mount the existing images/ folder:
        channels.append(
            {
                "ChannelName": "images",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/data/images/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        )

    return channels


def launch_training_job(
    sm_client,
    job_name: str,
    mode: str,
    bucket: str,
    role_arn: str,
    region: str,
    source_s3_uri: str,
    instance_type: str,
    volume_size_gb: int,
    max_run_seconds: int,
) -> str:
    """
    Start a SageMaker Training Job through boto3.
    SageMaker shuts the machine down automatically when the job finishes.
    """
    entry_script = {
        "train": "train.py",
        "eval": "eval.py",
        "download": "download_images.py",
    }[mode]

    image_uri = DLC_IMAGE_TEMPLATE.format(region=region)
    input_channels = build_input_channels(bucket=bucket, mode=mode)

    print(f"[2/3] Starting SageMaker job: {job_name}")
    print(f"      Mode: {mode} ({entry_script}) | Instance: {instance_type}")

    response = sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        InputDataConfig=input_channels,
        OutputDataConfig={
            "S3OutputPath": f"s3://{bucket}/sagemaker-output/",
        },
        ResourceConfig={
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": volume_size_gb,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": max_run_seconds,
        },
        Environment={
            "S3_BUCKET": bucket,
            "CONFIG_PATH": "config.yaml",
            "PYTHONUNBUFFERED": "1",
            # CT: model version (for the registry)
            "MODEL_VERSION": os.environ.get("MODEL_VERSION", "v0.0.0"),
            "GIT_COMMIT": os.environ.get("GIT_COMMIT", ""),
            "GIT_BRANCH": os.environ.get("GIT_BRANCH", ""),
        },
        HyperParameters={
            "sagemaker_program": f'"{entry_script}"',
            "sagemaker_submit_directory": f'"{source_s3_uri}"',
        },
    )
    return response["TrainingJobArn"]


def stream_job_logs_and_wait(sm_client, logs_client, job_name: str, poll_seconds: int = 15):
    """
    Print the SageMaker job status and CloudWatch logs live to the local screen.
    Exits the loop once the job finishes (Completed / Failed / Stopped).
    """
    print(
        f"[3/3] Live monitoring started ({job_name}). "
        f"Press Ctrl+C to stop watching (the job keeps running in the cloud).\n"
    )
    last_timestamp = 0
    seen_event_ids = set()
    last_status_line = ""

    while True:
        # 1. Check the job status
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        billable = desc.get("BillableTimeInSeconds", 0)

        status_line = (
            f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status} ({secondary}) "
            f"| Billable time: {billable}s"
        )
        if status_line != last_status_line:
            print(status_line)
            last_status_line = status_line

        # 2. Fetch CloudWatch logs and print the new lines
        try:
            log_resp = logs_client.filter_log_events(
                logGroupName="/aws/sagemaker/TrainingJobs",
                logStreamNamePrefix=job_name,
                startTime=last_timestamp,
            )
            for event in log_resp.get("events", []):
                event_id = event["eventId"]
                if event_id not in seen_event_ids:
                    seen_event_ids.add(event_id)
                    print("   ", event["message"].rstrip())
                    last_timestamp = max(last_timestamp, event["timestamp"])
        except logs_client.exceptions.ResourceNotFoundException:
            # While the machine is still starting (Starting/Downloading) the log group may not exist yet
            pass

        # 3. Break out of the loop once the job has finished
        if status in ("Completed", "Failed", "Stopped"):
            print("\n" + "=" * 60)
            print(f"Job finished! Final status: {status} | Total GPU time: {billable} seconds")
            if status == "Failed":
                print(f"Failure reason: {desc.get('FailureReason', 'Unknown')}")
            print("=" * 60)
            break

        time.sleep(poll_seconds)


def parse_args():
    parser = argparse.ArgumentParser(description="AWS SageMaker training/evaluation launcher")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "download"],
        default="train",
        help="Mode to run: train (train.py), eval (eval.py), download (download_images.py)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket name (or the S3_BUCKET environment variable)",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_ROLE_ARN"),
        help="SageMaker IAM role ARN (or the SAGEMAKER_ROLE_ARN environment variable)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="AWS region (default: us-west-2)",
    )
    parser.add_argument(
        "--instance-type",
        default=None,
        help=(
            "Instance type (picked automatically per mode when omitted: "
            "train->ml.g5.xlarge, eval->ml.g4dn.xlarge, download->ml.m5.xlarge)"
        ),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Custom job name (auto-generated from the timestamp when omitted)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="After starting the job, follow the CloudWatch logs live in the terminal",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.bucket or not args.role_arn:
        raise SystemExit(
            "Error: an S3 bucket and a SageMaker role ARN are required.\n"
            "Pass --bucket and --role-arn, or set the "
            "S3_BUCKET and SAGEMAKER_ROLE_ARN environment variables."
        )

    # Default instance type and disk size per mode
    default_instances = {
        "train": ("ml.g5.xlarge", 100, 28800),  # A10G 24GB GPU, 100GB disk, max 8 hours
        "eval": ("ml.g4dn.xlarge", 50, 3600),  # T4 16GB GPU, 50GB disk, max 1 hour
        "download": ("ml.m5.xlarge", 50, 7200),  # 4 vCPU, 50GB disk, max 2 hours
    }
    default_inst, volume_gb, max_run = default_instances[args.mode]
    instance_type = args.instance_type or default_inst

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"clip-{args.mode}-{timestamp}"

    # Create the boto3 clients
    session = boto3.Session(region_name=args.region)
    s3_client = session.client("s3")
    sm_client = session.client("sagemaker")
    logs_client = session.client("logs")

    # 1. Package the code in memory and upload it to S3
    s3_code_key = f"code/sourcedir_{job_name}.tar.gz"
    source_s3_uri = build_and_upload_source_archive(s3_client, args.bucket, s3_code_key)

    # 2. Start the SageMaker job
    job_arn = launch_training_job(
        sm_client=sm_client,
        job_name=job_name,
        mode=args.mode,
        bucket=args.bucket,
        role_arn=args.role_arn,
        region=args.region,
        source_s3_uri=source_s3_uri,
        instance_type=instance_type,
        volume_size_gb=volume_gb,
        max_run_seconds=max_run,
    )
    print(f"      Created successfully! ARN: {job_arn}\n")

    # 3. Follow the live logs if requested
    if args.follow:
        stream_job_logs_and_wait(sm_client, logs_client, job_name)


if __name__ == "__main__":
    main()
