import datetime


def get_version() -> str:
    return datetime.datetime.now().strftime("%Y.%m.%d")
