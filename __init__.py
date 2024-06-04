"""
basically just loguru configuration
"""

import sys
from loguru import logger

# Удаление стандартных обработчиков
logger.remove()

# Добавление нового обработчика с форматированием и поддержкой уровня INFO
logger.add(
    sys.stdout,
    format="<green>{time:MM-DD HH:mm}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)