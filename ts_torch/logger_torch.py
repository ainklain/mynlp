
import logging.handlers
from colorlog import ColoredFormatter

def get_level(level_str):
    level_dict = {'debug': logging.DEBUG,
                  'info': logging.INFO,
                  'warning': logging.WARNING,
                  'error': logging.ERROR,
                  'critical': logging.CRITICAL}
    return level_dict[level_str.lower()]


def logger(name, configs, filename='log', use_stream_handler=False):
    # 로거 & 포매터 & 핸들러 생성
    logger = logging.getLogger(name)
    if use_stream_handler:
        formatter_color = ColoredFormatter(configs.log_format,
                                           log_colors={
                                               'DEBUG': 'cyan',
                                               'INFO': 'white,bold',
                                               'WARNING': 'yellow',
                                               'ERROR': 'red,bold',
                                               'CRITICAL': 'red, bg_white',
                                           })
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter_color)
        logger.addHandler(stream_handler)

    formatter = logging.Formatter(configs.log_format)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=configs.log_filename(filename),
        maxBytes=configs.log_maxbytes,
        backupCount=configs.log_backupcount)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 로거 레벨 설정
    logger.setLevel(get_level(configs.log_level))

    return logger