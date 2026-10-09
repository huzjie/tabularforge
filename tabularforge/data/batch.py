"""Simple batching helper."""
import math


def batch_records(records, batch_size):
    """Yield consecutive batches of ``batch_size`` records."""
    for i in range(0, len(records), batch_size):
        yield records[i:i + batch_size]
