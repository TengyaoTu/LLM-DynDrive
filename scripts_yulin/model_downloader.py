from huggingface_hub import HfApi

HF_ENDPOINT = 'http://mirrors.tools.huawei.com/huggingface'
api = HfApi(endpoint=HF_ENDPOINT)
api.snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    repo_type="model",
    revision="main",
    local_dir="D:\\models\\",
    etag_timeout=10000
)