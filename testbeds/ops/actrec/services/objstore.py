"""S3 objects on SeaweedFS (class K: deterministic object key + `If-None-Match: *` conditional create)."""
from __future__ import annotations

import boto3
from botocore import handlers as _bh
from botocore.config import Config
from botocore.exceptions import ClientError

from ..config import S3_KEY, S3_SECRET, port
from .base import INFLIGHT_SOCKET_TIMEOUT_S, CallResult, call_with_deadline, nohook


def _client(route: str, read_timeout: float = INFLIGHT_SOCKET_TIMEOUT_S):
    c = boto3.client(
        "s3", endpoint_url=f"http://localhost:{port('s3', route)}", aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET, region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=read_timeout,
                      retries={"total_max_attempts": 1, "mode": "standard"}))
    # botocore sends PutObject with `Expect: 100-continue` and holds the body until the server answers. A
    # fault on the reply path then cuts the request before the body is sent, so "reply lost after the write"
    # cannot happen. Send headers and body together, as most HTTP clients do.
    c.meta.events.unregister("before-call.s3", _bh.add_expect_header)
    c.meta.events.unregister("before-call.s3.PutObject", _bh.add_expect_header)
    return c


class Objects:
    def __init__(self, bucket: str):
        self.bucket = bucket

    def ensure_bucket(self) -> None:
        s3 = _client("direct", 30)
        try:
            s3.create_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response["Error"].get("Code") not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise

    def put(self, key: str, body: bytes, tag: str | None = None, conditional: bool = True, route: str = "clean",
            hook=nohook, deadline_s: float = 4.0, repeat: int = 1) -> CallResult:
        kw = {"Bucket": self.bucket, "Key": key, "Body": body}
        if tag:
            kw["Metadata"] = {"op-id": tag}
        if conditional:
            kw["IfNoneMatch"] = "*"

        def fn():
            s3 = _client(route)
            s3.head_bucket(Bucket=self.bucket)      # warm the pooled connection
            hook()
            out = []
            for _ in range(repeat):
                try:
                    s3.put_object(**kw)
                    out.append("created")
                except ClientError as e:
                    code = e.response["ResponseMetadata"].get("HTTPStatusCode")
                    if code == 412 and repeat > 1 and out:
                        out.append("precondition_failed")   # the duplicate was refused by the key
                    else:
                        raise
            return {"results": out}
        return call_with_deadline(fn, deadline_s)

    # ---------------- reads ----------------
    def exists(self, key: str, route: str = "clean") -> int:
        s3 = _client(route, 30)
        try:
            s3.head_object(Bucket=self.bucket, Key=key)
            return 1
        except ClientError:
            return 0

    def metadata(self, key: str, route: str = "clean") -> dict:
        return _client(route, 30).head_object(Bucket=self.bucket, Key=key).get("Metadata", {})

    def truth_count(self, key: str) -> int:
        return self.exists(key, route="direct")
