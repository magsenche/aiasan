import logging

import colorlog



def logger(name, level=logging.INFO):
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message_log_color)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "blue",
                "INFO": "reset",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            }
        },
        style="%",
    )

    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log
