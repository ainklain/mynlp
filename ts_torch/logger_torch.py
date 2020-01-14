import logging.handlers


def get_level(level_str):
    level_dict = {'debug': logging.DEBUG,
                  'info': logging.INFO,
                  'warning': logging.WARNING,
                  'error': logging.ERROR,
                  'critical': logging.CRITICAL}
    return level_dict[level_str.lower()]


def logger(name, configs):
    # 로거 & 포매터 & 핸들러 생성
    logger = logging.getLogger(name)
    formatter = logging.Formatter(configs.log_format)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.RotatingFileHandler(
        filename=configs.log_filename,
        maxBytes=configs.log_maxbytes,
        backupCount=configs.log_backupcount)

    # 핸들러 & 포매터 결합
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # 로거 & 핸들러 결합
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    # 로거 레벨 설정
    logger.setLevel(get_level(configs.log_level))

    return logger