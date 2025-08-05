import sys
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def div_num(a,b):
    try:
        x = a/b
        logger.info("Division between two variables")
        return x
    except Exception as e:
        logger.error("Error occured in division.")
        raise CustomException("Zero Division Error: (Denomenator != 0)", e)

if __name__=="__main__":
    try:
        logger.info("Test Started>>>")
        div_num(2,0)
    except CustomException as e:
        logger.error(str(e))

