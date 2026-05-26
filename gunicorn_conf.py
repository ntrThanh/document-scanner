import multiprocessing
import os


bind = os.getenv("BIND", "0.0.0.0:8888")
worker_class = "uvicorn.workers.UvicornWorker"

# The app keeps upload/output/job state in process memory. Keep one process so
# browser polling always sees the same queue, then tune in-app concurrency with
# SCAN_MAX_CONCURRENT_JOBS.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "7200"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "60"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))
worker_tmp_dir = os.getenv("GUNICORN_WORKER_TMP_DIR", "/dev/shm")

accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "500"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))


def when_ready(server):
    server.log.info(
        "Serving with %s worker(s), %s thread(s), cpu_count=%s, "
        "SCAN_MAX_CONCURRENT_JOBS=%s, SCAN_MAX_CONCURRENT_OCR=%s",
        workers,
        threads,
        multiprocessing.cpu_count(),
        os.getenv("SCAN_MAX_CONCURRENT_JOBS", "2"),
        os.getenv("SCAN_MAX_CONCURRENT_OCR", "1"),
    )
