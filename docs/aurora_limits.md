# Aurora Cloud Storage — Limits & Quotas

- Maximum single file size: 5 GB on Starter/Pro, 5 TB on Enterprise (via multipart upload).
- API rate limit: 100 requests/minute on Starter, 1,000/minute on Pro, negotiable on Enterprise.
- Maximum objects per bucket: no hard limit, but list operations paginate above 100,000 objects.
- Webhook delivery retries up to 5 times with exponential backoff before an event is dropped.
- Account-level bucket limit: 10 on Starter, 100 on Pro, unlimited on Enterprise.

Aurora does not publish real-time infrastructure status metrics (CPU/memory of underlying nodes) to
customers; only the public status page (uptime/incidents) is available.
