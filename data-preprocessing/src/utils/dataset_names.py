"""Dataset and split naming helpers for file-based scripts."""

from pathlib import Path

SPLIT_SUFFIXES = (
    '.train_all',
    '.dev_all',
    '.test_all',
    '.train_smallpp',
    '.dev_smallpp',
    '.test_smallpp',
    '.train_medium',
    '.dev_medium',
    '.test_medium',
    '.train',
    '.dev',
    '.test',
)


def extract_dataset_name_from_filename(filename: str, *, no_group: bool = False) -> str:
    """
    Extract dataset name from ``.jsonl.gz`` filename.

    :param filename: File name
    :param no_group: Keep split suffixes when True
    :return: Dataset name
    """
    name = filename
    if name.endswith('.gz'):
        name = name[:-3]
    if name.endswith('.jsonl'):
        name = name[:-6]
    name = name.replace('.analysis', '')
    if no_group:
        return name
    for suffix in SPLIT_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def extract_dataset_name_from_path(filepath: Path) -> str | None:
    """
    Extract dataset name from ``.analysis.jsonl`` path variants.

    :param filepath: File path
    :return: Dataset name or ``None`` when empty
    """
    name = filepath.name
    if name.endswith('.gz'):
        name = name[:-3]
    if name.endswith('.tsv'):
        name = name[:-4]
    name = name.replace('.analysis.jsonl', '')
    for suffix in SPLIT_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name or None


def detect_split_type(filename: str, *, default: str = 'train') -> str:
    """
    Detect split token from file name.

    :param filename: File name
    :param default: Fallback split when no token is present
    :return: ``train``, ``dev`` or ``test``
    """
    if '.train' in filename or filename.startswith('train.'):
        return 'train'
    if '.dev' in filename or filename.startswith('dev.'):
        return 'dev'
    if '.test' in filename or filename.startswith('test.'):
        return 'test'
    return default

