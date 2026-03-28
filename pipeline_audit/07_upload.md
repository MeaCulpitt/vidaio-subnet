# 07 — Upload

## Current Implementation

**File:** `/root/vidaio-subnet/services/upscaling/server.py` (lines 829-848)

```python
await storage_client.upload_file(object_name, processed_video_path)
sharing_link = await storage_client.get_presigned_url(object_name)
schedule_file_deletion(object_name)  # 10-min auto-delete
```

## Baseline Timing

Estimated 2-5s depending on file size and network. 4K HEVC CQ=20 files are
typically 5-20 MB for 10s content.

## Improvement Opportunities

| # | Improvement | Est. Impact | Ease |
|---|-------------|-------------|------|
| 1 | Multipart upload for large files | ~1-2s faster | medium |
| 2 | Start upload while encoding last frames | ~1s overlap | hard |
| 3 | Ensure S3 region matches miner | reduce latency | easy |

## Status: [ ] pending
