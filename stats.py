import csv
from jtop import jtop, JtopException
import time


if __name__ == "__main__":
    jetson = jtop()
    jetson.start()
    
    LOGGING_TIME = 30
    EXECUTION_TIME = 3600
    CSV_PATH = "stats.csv"

    start_time = time.time()

    with jtop() as jetson:
        # Make csv file and setup csv
        with open(CSV_PATH, "a") as csvfile:
            stats = jetson.stats
            # Initialize cws writer
            writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
            # Write header
            writer.writeheader()
            # Write first row
            writer.writerow(stats)
            # Start loop
            while jetson.ok() and (time.time() - start_time) < EXECUTION_TIME:
                print(time.time() - start_time)
                time.sleep(LOGGING_TIME)
                stats = jetson.stats
                # Write row
                writer.writerow(stats)
                print("Log at {time}".format(time=stats['time']))
        csvfile.close()

    print("Done !")









