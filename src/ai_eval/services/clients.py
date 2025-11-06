from google.cloud import storage
from ai_eval.config import global_config as glob


class GCPClient:
    def __init__(self, bucket_name: str):
        """Initialize GCP Storage Client with a specific bucket."""
        self.bucket_name = bucket_name
        self.client = storage.Client(project=glob.GCP_PROJECT)
        self.bucket = self.client.bucket(bucket_name)

    def get_blob(self, blob_name: str) -> storage.Blob:
        """Instantiates a blob from the bucket."""
        return self.bucket.blob(blob_name)

    # def download_blob_bytes(self, blob_name: str) -> bytes:
    #     """Download the content of a blob."""
    #     blob = self.get_blob(blob_name)
    #     return blob.download_as_bytes()

    # def upload_blob_bytes(self, blob_name: str, data: bytes) -> None:
    #     """Upload data to a blob."""
    #     blob = self.get_blob(blob_name)
    #     blob.upload_from_string(data)
