import os
import shutil
from os.path import join, isdir, exists
from os import listdir


cur_sep = os.path.sep


class DiffCollect:

    def __init__(self, src_path, dst_path):
        self.src = src_path
        self.dst = dst_path
        self.count = 0
        if not exists(self.dst):
            os.makedirs(self.dst)

    def collect(self, path):
        items = listdir(path)
        for item in items:
            p = join(path, item)
            if isdir(p):
                self.collect(p)
            if not p.endswith(".diff"):
                continue
            # 提取前缀信息和文件名称
            filenames = p.replace(self.src, "").split(cur_sep)
            prefix = ""
            for i in range(len(filenames)-1):
                prefix += filenames[i] + '#'
            filename = filenames[-1]
            filename = prefix + filename
            save = join(self.dst, filename)
            shutil.copyfile(p, save)
            self.count += 1


if __name__ == '__main__':
    diffCollect = DiffCollect(os.getcwd(), join(os.getcwd(), "DiffFiles"))
    diffCollect.collect(diffCollect.src)
    print(f"Over, ... collected {diffCollect.count} files.")
