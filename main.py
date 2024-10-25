import multiprocessing
import subprocess


def run_script1():
    subprocess.run(["python3", "object-color-detection.py"])


def run_script2():
    subprocess.run(["python3", "/home/pi/Desktop/stormines/control-command.py"])


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_script1)
    p2 = multiprocessing.Process(target=run_script2)

    p1.start()
    p2.start()

    p1.join()
    p2.join()