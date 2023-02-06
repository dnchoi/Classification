# -*- coding: utf-8 -*-
import logging.handlers
import os
from colorlog import ColoredFormatter


class Setting:
    """로거 세팅 클래스

    ::

            Setting.LEVEL = logging.INFO # INFO 이상만 로그를 작성
    """

    LEVEL = logging.DEBUG
    # LEVEL         = logging.WARNING
    # LEVEL         = logging.ERROR
    FILENAME = "history.log"
    MAX_BYTES = 15 * 1024 * 1024
    BACKUP_COUNT = 100
    FORMAT = "%(asctime)s[%(levelname)s|%(name)s,%(lineno)s] %(message)s"


def Logger(name, save):
    """파일 로그 클래스

    :param name: 로그 이름
    :type name: str
    :return: 로거 인스턴스

    ::

            logger = Logger(__name__)
            logger.info('info 입니다')
    """

    from os.path import expanduser

    home = expanduser(".")

    path_only = name.split("/")[0]
    if not os.path.isdir(path_only):
        os.mkdir(path_only)
        print(f"create log directory : {path_only}")

    # 로거 & 포매터 & 핸들러 생성
    logger = logging.getLogger(name)
    # formatter = logging.Formatter(Setting.FORMAT)
    streamHandler = logging.StreamHandler()
    if save:
        fileHandler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(home, name + "__" + Setting.FILENAME),
            maxBytes=Setting.MAX_BYTES,
            backupCount=Setting.BACKUP_COUNT,
        )

    color_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s[%(levelname)s|%(name)s,%(lineno)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "blue,bold",
            "INFO": "green,bold",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    # 핸들러 & 포매터 결합
    streamHandler.setFormatter(color_formatter)
    # 로거 & 핸들러 결합
    logger.addHandler(streamHandler)

    if save:
        fileHandler.setFormatter(color_formatter)
        logger.addHandler(fileHandler)

    # 로거 레벨 설정
    logger.setLevel(Setting.LEVEL)

    return logger


class LoggerWraper:
    def __init__(self) -> None:
        self.logger = None

    def get(self):
        return self.logger

    def make_logger(self, topic):
        self.logger = Logger(topic)
        return self.logger
