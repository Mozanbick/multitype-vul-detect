import pickle
from os.path import join


def parsePickle(basedir, filename):
    f = open(join(basedir, filename), "rb")
    res = pickle.load(f)
    fw = open(join(basedir, "sensi_funcs.txt"), "w")
    fw.write(str(res))
    print("over...")


if __name__ == '__main__':
    baseDir = "./sundries/"
    fileName = "sensitive_func.pkl"
    parsePickle(baseDir, fileName)
