import base64
import functools
import io
import logging
from typing import Any, Callable, Optional

import colorlog
import streamlit as st
from crewai_tools import Tool, tool
from streamlit.runtime.uploaded_file_manager import UploadedFile


def logger(name: str, level=logging.INFO) -> logging.Logger:
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


def partial(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        new_func = functools.partial(func, *args, **kwargs)
        new_func.__name__ = getattr(func, "__name__", "")
        new_func.__doc__ = getattr(func, "__doc__", "")
        annotations = {}
        for name, annotation in getattr(func, "__annotations__", {}).items():
            if name not in kwargs:
                annotations[name] = annotation
        new_func.__annotations__ = annotations
        return new_func

    return wrapper


def get_crewai_tool(
    tool_func: Callable, tool_name: Optional[str] = None, **kwargs: Any
) -> Tool:
    partial_tool_func = partial(tool_func)(**kwargs)
    return tool(tool_name)(partial_tool_func)


def display(file: UploadedFile):
    if file.type == "application/pdf":
        display_pdf(file)
    elif file.type == "application/octet-stream":
        display_md(file)


def display_md(file: io.BytesIO):
    markdown_content = file.read().decode("utf-8")
    st.markdown(markdown_content, unsafe_allow_html=True)


def display_pdf(file: io.BytesIO):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)
