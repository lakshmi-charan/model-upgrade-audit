"""Thin adapters over the real services. Agent-facing writes go through the clean or fault proxy;
`truth_*` functions are grader-side and always use the direct ports.

Client-library retries are switched OFF everywhere (boto3 total_max_attempts=1, redis-py Retry(0)),
because a hidden library retry would silently turn a lost acknowledgement into a duplicate. The SDK-retry
baseline re-enables retries explicitly. Exception: the TigerBeetle client always retries internally and
deduplicates by session; that is part of its native-key contract and is recorded as such.
"""
