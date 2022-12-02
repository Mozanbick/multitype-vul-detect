"""
version: 2022/11/25
author: zjh

This script is going to analyze several statistics (e.g. number of CWEs, number of files) of datasets
"""

import sys
import os
import glob


def _count_files_diff(base_dir: str):
    count = 0
    path = base_dir + '/*/*.diff'
    if 'FAN' in base_dir:
        path = base_dir + '/*/*/*.diff'
    for _ in glob.glob(path):
        count += 1
    return count


def get_CWEs(base_dirs: list[str]):
    """
    Args:
        base_dirs: base directories to traverse
    """
    if not base_dirs:
        return {}
    cwe_dict = {}
    for root in base_dirs:  # first traverse: Directories
        for tp in os.listdir(root):
            d = os.path.join(root, tp)
            if not os.path.isdir(d):
                continue
            count = 0
            for name in os.listdir(d):
                if name.startswith('CWE'):
                    if name not in cwe_dict:
                        cwe_dict[name] = 0
                    cwe_dict[name] += _count_files_diff(d + '/' + name)
                    count += 1
            if os.listdir(d) and count == 0:
                raise ValueError('Base dir error: need to be `base_dir/cew_id/cve_id/files`')
    return cwe_dict


def output(out_dict, save_dir=''):
    save_path = os.path.join(save_dir, "output.txt")
    out_list = []
    with open(save_path, "w") as fp:
        for cwe, num in out_dict.items():
            out_list.append((cwe, num))
        out_list.sort(key=lambda x: x[1], reverse=True)
        for item in out_list:
            fp.write(f"{item[0]}: {item[1]}\n")


if __name__ == '__main__':
    option = sys.argv[1]
    dirs = sys.argv[2:]
    if option == 'cwe':
        output(get_CWEs(dirs))
