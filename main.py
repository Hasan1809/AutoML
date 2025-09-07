from src.logger import logging
from src.exception import AutoMLException
import sys
if __name__ == "__main__":
    logging.info("testing logging module")
    try:
        a = 20/0
    except Exception as e:
        raise AutoMLException(e, sys)