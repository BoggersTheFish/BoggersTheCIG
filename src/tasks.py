"""
Celery tasks for Full-TS background processing.
Run: celery -A src.tasks worker --loglevel=info
"""
from celery import Celery
from src.config import REDIS_URL
from src.continuous_thinker import run_loop_iteration

app = Celery("full_ts", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@app.task
def think_loop_iteration():
    """Single thinking loop iteration (async)."""
    return run_loop_iteration()
