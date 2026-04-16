import time
import logging

logger = logging.getLogger(__name__)


class DebuggingClock:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        logger.info(self)

    def elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError("Clock has not been started and stopped properly.")
        return self.end_time - self.start_time

    def __str__(self):
        return f"Elapsed time: {self.elapsed_time():.4f} seconds"
