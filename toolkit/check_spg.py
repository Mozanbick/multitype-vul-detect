import pickle
from os.path import join


def read_fpg(spgs_dir: str, group: str):
    save_path = join(spgs_dir, f"spg_list_{group}.pkl")
    with open(save_path, "rb") as fp:
        fpg_list = pickle.load(fp)
    with open("./spgs_info.txt", "w") as fp:
        for g in fpg_list:
            print(g, file=fp)
            fp.write('----------------------------\n\n')


if __name__ == '__main__':
    fd = "../joern/repository/sard/spgs"
    group = "group6"
    read_fpg(fd, group)
