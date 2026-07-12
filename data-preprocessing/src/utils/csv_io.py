"""CSV helper utilities."""

import csv
import sys


def ensure_large_csv_fields(preferred_limit: int = sys.maxsize) -> None:
    """
    Raise CSV field-size limit with overflow-safe fallback.

    :param preferred_limit: Requested field-size limit
    """
    limit = preferred_limit
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
            if limit <= 0:
                csv.field_size_limit(2**31 - 1)
                return

