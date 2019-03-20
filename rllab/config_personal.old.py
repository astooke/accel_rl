USE_GPU = True
KUBE_PREFIX = "rein_"
DOCKER_LOG_DIR = "/tmp/expt"
AWS_S3_PATH = "s3://openai-kubernetes-sci-rein/exp2017-b"
AWS_S3_BUCKET = "s3://openai-kubernetes-sci-rein"
AWS_IMAGE_ID = "ami-e32fff8d"
if USE_GPU:
    AWS_INSTANCE_TYPE = "p2.xlarge"
    DOCKER_IMAGE = "rein/rllab-exp-gpu-tf:b"
    # USE_TF = True
    # DOCKER_IMAGE = "neocxi/rllab_exp_gpu_tf"
    # DOCKERFILE_PATH = "/Users/rein/programming/workspace_py/rllab-private/docker/gpu_Dockerfile"
    # Leave some room for background processes.
    KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 3.5,
        },
        "limits": {
            "cpu": 3.5,
        }
    }
else:
    AWS_INSTANCE_TYPE = "m4.4xlarge"
    DOCKER_IMAGE = "rein/rllab-exp-gpu-tf:b"
    DOCKERFILE_PATH = "/Users/rein/programming/workspace_py/rllab-private/docker/Dockerfile"
    # Leave some room for background processes.
    KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 3.5,
        },
        "limits": {
            "cpu": 3.5,
        }
    }
AWS_KEY_NAME = "ap-northeast-2"
AWS_SPOT = False
AWS_SPOT_PRICE = '10.0'
AWS_ACCESS_KEY = 'AKIAIRE2HOZRAM4X4EPA'
AWS_ACCESS_SECRET = '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'
AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"
AWS_SECURITY_GROUPS = ["rllab"]
AWS_REGION_NAME = "us-west-2"
AWS_CODE_SYNC_S3_PATH = "s3://openai-kubernetes-sci-rein/code2"
CODE_SYNC_IGNORES = ["*.git/*", "*data/*", "*src/*",
                     "*.pods/*", "*tests/*", "*examples/*", "docs/*"]
LOCAL_CODE_DIR = "/home/rein/workspace_python/rllab"
LABEL = "rein"
DOCKER_CODE_DIR = "/root/code/rllab"
MUJOCO_KEY_PATH = "/home/rein/.mujoco/"

KUBE_DEFAULT_NODE_SELECTOR = {
    "aws/type": AWS_INSTANCE_TYPE,
}

ALL_REGION_AWS_IMAGE_IDS = {
    # "ap-northeast-1": "ami-673e4500",
    # "ap-northeast-2": "ami-27f12049",
    # "ap-south-1":     "ami-75c8be1a",
    # "ap-southeast-1": "ami-1c29837f",
    # "ap-southeast-2": "ami-03717660",
    # "eu-central-1":   "ami-374a8558",
    # "eu-west-1":      "ami-3dcd915b",
    # "sa-east-1":      "ami-237d184f",
    # "us-east-1":      "ami-3f6d9929",
    # "us-east-2":      "ami-73527716",
    # "us-west-1":      "ami-e46e3c84",
    # "us-west-2":      "ami-cfd967af",
    # "us-west-2": "ami-3f088d5f",
    # "us-west-1": "ami-1df4d07b",
    # "ap-northeast-2": "ami-929242fc",

    # rocky's ami
    "ap-northeast-2": "ami-e32fff8d",
    "ap-south-1": "ami-0d0a7a62",
    "ap-southeast-1": "ami-1ac97979",
    "ap-southeast-2": "ami-55868536",
    "eu-central-1": "ami-6039ec0f",
    "eu-west-1": "ami-e9bd938f",
    "sa-east-1": "ami-a898fec4",
    "us-east-1": "ami-f33ee7e5",
    "us-east-2": "ami-4082a725",
    "us-west-1": "ami-3faef05f",
    "us-west-2": "ami-7576f415",

    # gpu working
    # ap-northeast-1: ami-a6b5f6c1
    # ap-northeast-2: ami-929242fc
    # ap-south-1: ami-caa4d5a5
    # ap-southeast-1: ami-59e85f3a
    # ap-southeast-2: ami-2e70714d
    # eu-central-1: ami-6d22ea02
    # eu-west-1: ami-1df4d07b
    # sa-east-1: ami-7c284f10
    # us-east-1: ami-4d7ebf5b
    # us-east-2: ami-50e3c635
    # us-west-2: ami-3f088d5f
}
