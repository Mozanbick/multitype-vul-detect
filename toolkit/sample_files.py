import os
import random


if __name__ == '__main__':
    max_len = 10000
    cwd = os.getcwd()
    target = os.path.join(cwd, "group1")
    files = os.listdir(target)
    random.shuffle(files)
    preserve = files[:max_len]
    for item in files:
        if item in preserve:
            continue
        path = os.path.join(target, item)
        if os.path.exists(path):
            os.remove(path)
