from jtop import jtop, JtopException
import logging
import os
import time

def load_logger(logger_path):
    # create logger with 'spam_application'
    logger = logging.getLogger('Lux')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger

logger = load_logger("stats.log")

if __name__ == "__main__":
    jetson = jtop()
    jetson.start()

    LOGGING_TIME = 30
    EXECUTION_TIME = 3600

    start_time = time.time()

    with jtop() as jetson:
        # Make csv file and setup csv
            stats = jetson.stats
            # Start loop
            while jetson.ok() and (time.time() - start_time) < EXECUTION_TIME:
                time.sleep(LOGGING_TIME)
                stats = jetson.stats
                # Log
                print("Log at {time}".format(time=stats['time']))
                del stats["time"]
                del stats["uptime"]
                logger.info(stats)

    print("Done !")









