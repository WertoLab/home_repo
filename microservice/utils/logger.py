from loguru import logger


logger.remove()

logger.add(
    "logs/validation_errors.log",
    format="{time} | {level} | {message}",
    level="ERROR",
    rotation="1 day",
    compression="zip",
    filter=lambda record: record["extra"]["name"] == "validation_error_log",
)


logger.add(
    "logs/internal_errors.log",
    format="{time} | {level} | {message}",
    level="ERROR",
    rotation="1 day",
    compression="zip",
    filter=lambda record: record["extra"]["name"] == "internal_errors_log",
)

internal_error_logger = logger.bind(name="internal_errors_log")
validation_error_logger = logger.bind(name="validation_error_log")
