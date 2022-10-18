import pickle
from os.path import join


def read_fpg(fpgs_dir: str, group: str):
    save_path = join(fpgs_dir, f"fpg_list_{group}.pkl")
    with open(save_path, "rb") as fp:
        fpg_list = pickle.load(fp)
    with open("./fpgs_info.txt", "w") as fp:
        for g in fpg_list:
            print(g, file=fp)
            fp.write('----------------------------\n\n')


if __name__ == '__main__':
    fd = "joern/repository/newnvd/fpgs"
    group = "group0"
    read_fpg(fd, group)
