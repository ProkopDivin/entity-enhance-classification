"""Date parsing helpers shared across CLI scripts."""

from datetime import datetime, timezone


def parse_iso_or_ymd_naive(date_str: str | None) -> datetime | None:
    """
    Parse ISO or ``YYYY-MM-DD`` date string to timezone-naive ``datetime``.

    :param date_str: Date string
    :return: Parsed date or ``None`` if invalid
    """
    if not date_str:
        return None

    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None


def parse_iso_to_aware_utc(date_str: str | None) -> datetime | None:
    """
    Parse ISO date string to timezone-aware ``datetime``.

    :param date_str: Date string
    :return: Parsed date or ``None`` if invalid
    """
    if not date_str:
        return None
    try:
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        else:
            date_str = date_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def parse_ymd_or_none(date_str: str | None) -> datetime | None:
    """
    Parse ``YYYY-MM-DD`` date string, treating ``None`` string as missing.

    :param date_str: Date string
    :return: Parsed date or ``None`` if invalid/missing
    """
    if date_str == 'None' or not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

