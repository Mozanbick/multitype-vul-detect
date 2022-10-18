import pickle
import os
import re
from os import listdir
from os.path import join, isdir, exists
from typing import Dict, List


class DiffParser:

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.vul_dict: Dict[str, Dict[str, List]] = {}

    @staticmethod
    def _parse_diff_file_nvd(path: str):
        with open(path, "r") as fd:
            contents = fd.readlines()
        c_dict = {}
        cur_stat = []
        sus_stat = []
        has_minus = False
        cur_name = ''
        file_name = ''
        for item in contents:
            if item.startswith('diff '):
                file_name = item.strip().split('/')[-1].strip()
            elif item.startswith('diff') or item.startswith('index') or \
                    item.startswith('---') or item.startswith('+++') or \
                    item.startswith('+'):
                continue
            elif item.startswith('@@'):  # start line of each patch block in the diff file
                if cur_name:
                    if has_minus:
                        c_dict[cur_name] += cur_stat
                    else:
                        c_dict[cur_name] += sus_stat
                    cur_stat = []
                    sus_stat = []
                    has_minus = False
                pt = re.compile(r'-([0-9]+),([0-9]+) (\+[0-9]+),([0-9]+)')
                ret = re.search(pt, item)
                line = item[ret.start():ret.end()]
                cur_line = int(line.split(' ')[0].split(',')[0].removeprefix('-'))
                try:
                    func_name = item.split('@@')[-1].strip()
                    func_name = func_name[:func_name.index('(')].strip().split(' ')[-1].strip()
                except:
                    func_name = item.split('@@')[-1].strip()
                cur_name = func_name
                if cur_name not in c_dict:
                    c_dict[cur_name] = []
            elif item.startswith('-') and not item.startswith('---'):
                # delete lines in patch, usually corresponding to vulnerabilities
                stat = item.removeprefix('-').strip()
                if stat == '{' or stat == '}':
                    continue
                cur_stat.append(stat)
                has_minus = True
            else:  # origin statement in OLD files
                stat = item.strip()
                if stat == '{' or stat == '}':
                    continue
                sus_stat.append(stat)
        if cur_name:
            if has_minus:
                c_dict[cur_name] += cur_stat
            else:
                c_dict[cur_name] += sus_stat
        return file_name, c_dict

    def _parse_nvd(self, path: str):
        items = listdir(path)
        for item in items:
            p = join(path, item)
            if isdir(p):
                self._parse_nvd(p)
            if not p.endswith('.diff'):
                continue
            prefix = item.removesuffix('.diff')
            targets = []
            for _ in items:
                if _.endswith('OLD.c') and prefix in _:
                    targets.append(_)
            file_name, c_dict = self._parse_diff_file_nvd(p)
            title = p.removeprefix(self.base_dir)
            names = title.split(os.sep)
            testID = ''
            for _ in range(len(names) - 1):
                testID += names[_] + '#'
            if testID not in self.vul_dict:
                self.vul_dict[testID] = {}
            for file in targets:
                func = file.removeprefix(prefix+'_').removesuffix('_OLD.c').strip()
                if file_name in func:
                    func = func.removeprefix(file_name+'_')
                vul_line = []
                stats = c_dict[func] if func in c_dict else []
                if not stats:
                    continue
                with open(join(path, file)) as fs:
                    lines = fs.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    cur_stat = stats[0]
                    if line == cur_stat:
                        vul_line.append(idx+1)
                        stats.remove(cur_stat)
                        if not stats:
                            break
                self.vul_dict[testID][file] = vul_line

    def _parse_new_nvd(self, src_path: str, diff_path: str):
        items = listdir(diff_path)
        for item in items:
            sp = join(src_path, item)
            dp = join(diff_path, item)
            if isdir(dp):
                self._parse_new_nvd(sp, dp)
            if not dp.endswith('.diff'):
                continue
            if not exists(src_path):
                continue
            prefix = item.removesuffix('.diff')
            targets = []
            for _ in listdir(src_path):
                if _.endswith('OLD.c') and prefix in _:
                    targets.append(_)
            file_name, c_dict = self._parse_diff_file_nvd(dp)
            names = diff_path.split(os.sep)
            cve = [i for i, s in enumerate(names) if s.startswith('CVE')]  # one item
            testID = names[cve[0] - 1] + '#' + names[cve[0]]
            if testID not in self.vul_dict:
                self.vul_dict[testID] = {}
            for file in targets:
                func = file.removeprefix(prefix+'_').removesuffix('_OLD.c').strip()
                if file_name in func:
                    func = func.removeprefix(file_name+'_')
                vul_line = []
                stats = c_dict[func] if func in c_dict else []
                if not stats:
                    continue
                with open(join(src_path, file)) as fs:
                    lines = fs.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    cur_stat = stats[0]
                    if line == cur_stat:
                        vul_line.append(idx+1)
                        stats.remove(cur_stat)
                        if not stats:
                            break
                self.vul_dict[testID][file] = vul_line

    @staticmethod
    def _parse_diff_file_fan(path: str):
        with open(path, "r") as fd:
            contents = fd.readlines()
        c_dict = {}
        cur_stat = []
        sus_stat = []
        has_minus = False
        cur_name = ''
        for item in contents:
            if item.startswith('diff') or item.startswith('index') or \
                    item.startswith('---') or item.startswith('+++') or \
                    item.startswith('+'):
                continue
            elif item.startswith('@@'):
                if cur_name:
                    if has_minus:
                        c_dict[cur_name] += cur_stat
                    else:
                        c_dict[cur_name] += sus_stat
                    cur_stat = []
                    sus_stat = []
                    has_minus = False
                pt = re.compile(r'-([0-9]+),([0-9]+) (\+[0-9]+),([0-9]+)')
                ret = re.search(pt, item)
                line = item[ret.start():ret.end()]
                cur_line = int(line.split(' ')[0].split(',')[0].removeprefix('-'))
                func_info = item.split('@@')[-1].strip()
                if not func_info:
                    func_info = '-'
                cur_name = func_info
                if cur_name not in c_dict:
                    c_dict[cur_name] = []
            elif item.startswith('-'):
                stat = item.removeprefix('-').strip()
                if stat == '{' or stat == '}':
                    continue
                cur_stat.append(stat)
            else:
                stat = item.strip()
                if stat == '{' or stat == '}':
                    continue
                sus_stat.append(stat)
        if cur_stat and cur_name:
            if has_minus:
                c_dict[cur_name] += cur_stat
            else:
                c_dict[cur_name] += sus_stat
        return c_dict

    def _parse_fan(self, path: str):
        items = listdir(path)
        for item in items:
            p = join(path, item)
            if isdir(p):
                self._parse_fan(p)
            if not p.endswith('.diff'):
                continue
            targets = []
            for _ in items:
                if _.endswith('before.c'):
                    targets.append(_)
            c_dict = self._parse_diff_file_fan(p)
            title = p.removeprefix(self.base_dir)
            names = title.split(os.sep)
            testID = ''
            for _ in range(len(names) - 1):
                testID += names[_] + '#'
            if testID not in self.vul_dict:
                self.vul_dict[testID] = {}
            for file in targets:
                vul_line = []
                with open(join(path, file)) as fs:
                    lines = fs.readlines()
                # first line, func name
                func = lines[0].strip()
                stats = c_dict[func] if func in c_dict else []
                if not stats:
                    continue
                filename = testID + '_' + file
                for idx, line in enumerate(lines):
                    line = line.strip()
                    cur_stat = stats[0]
                    if line == cur_stat:
                        vul_line.append(idx+1)
                        stats.remove(cur_stat)
                        if not stats:
                            break
                self.vul_dict[testID][filename] = vul_line

    @staticmethod
    def _parse_diff_file_old_nvd(path: str):
        with open(path, "r") as fd:
            contents = fd.readlines()
        c_dict = []
        cur_stat = []
        sus_stat = []
        has_minus = False
        cur_name = ''
        for item in contents:
            if item.startswith('diff') or item.startswith('index') or \
                    item.startswith('---') or item.startswith('+++') or \
                    item.startswith('+'):
                continue
            elif item.startswith('@@'):
                if cur_stat or sus_stat:
                    if has_minus:
                        c_dict.append(cur_stat)
                    else:
                        c_dict.append(sus_stat)
                cur_stat = []
                sus_stat = []
                has_minus = False
            elif item.startswith('-'):
                stat = item.removeprefix('-').strip()
                if stat == '{' or stat == '}' or stat == '':
                    continue
                cur_stat.append(stat)
                has_minus = True
            else:
                stat = item.strip()
                if stat == '{' or stat == '}' or stat == '':
                    continue
                sus_stat.append(stat)
        if cur_stat or sus_stat:
            if has_minus:
                c_dict.append(cur_stat)
            else:
                c_dict.append(sus_stat)
        return c_dict

    @staticmethod
    def _check_match(src: List[str], dst: List[str]):
        if not src:
            return []
        match = True
        idx = 0
        sus_line = []
        try:
            for line in dst:
                while src[idx] == '{' or src[idx] == '}' or src[idx] == '':
                    idx += 1
                if not src[idx] == line:
                    match = False
                    break
                else:
                    sus_line.append(idx)
                    idx += 1
        except IndexError:
            return []
        if match:
            return sus_line
        else:
            return []

    def _parse_old_nvd_sub(self, path: str, src_path: str, src_dict: Dict[str, List[str]]):
        items = listdir(path)
        for item in items:
            p = join(path, item)
            if isdir(p):
                self._parse_old_nvd_sub(p, src_path, src_dict)
            if not p.endswith('.txt'):
                continue
            testID = item.removesuffix('.txt')
            if testID not in src_dict:
                continue
            c_dict = self._parse_diff_file_old_nvd(p)
            files = src_dict[testID]
            if testID not in self.vul_dict:
                self.vul_dict[testID] = {}
            for file in files:
                vul_line = []
                with open(join(src_path, file), "r") as fp:
                    lines = fp.readlines()
                lines = list(map(lambda x: x.strip(), lines))
                for stats in c_dict:
                    if not stats:
                        continue
                    try:
                        ids = [i for i, l in enumerate(lines) if l == stats[0]]
                        for idx in ids:
                            sus_line = self._check_match(lines[idx:], stats)
                            vul_line += list(map(lambda x: x + idx + 1, sus_line))
                    except ValueError:
                        continue
                self.vul_dict[testID][file] = list(set(vul_line))

    def _parse_old_nvd(self, src_path: str, diff_path: str):
        items = listdir(src_path)
        src_dict = {}
        for item in items:
            testID = item.split('@@')[0]
            label = int(item.split('@@')[1])
            if label == 0:
                continue
            if testID in src_dict:
                src_dict[testID].append(item)
            else:
                src_dict[testID] = [item]
        self._parse_old_nvd_sub(diff_path, src_path, src_dict)

    def parse(self, dataset='nvd', src_dir='', diff_dir=''):
        if dataset == 'nvd':
            self._parse_nvd(self.base_dir)
        elif dataset == 'fan':
            self._parse_fan(self.base_dir)
        elif dataset == 'oldnvd':
            self._parse_old_nvd(src_dir, diff_dir)
        elif dataset == 'newnvd':
            self._parse_new_nvd(src_dir, diff_dir)
        self.save()

    def save_to_txt(self):
        with open(join(self.base_dir, "vul_line.txt"), "w") as fp:
            fp.write(str(self.vul_dict))
            fp.write('\n\n')
            fp.write(f"Total len: {len(self.vul_dict)}\n")

    def save(self):
        with open(join(self.base_dir, "vul_line.pkl"), "wb") as fp:
            pickle.dump(self.vul_dict, fp)


if __name__ == '__main__':
    base_dir = "../sundries/newnvd/"
    diffParser = DiffParser(os.getcwd())
    diffParser.parse('newnvd', join(os.getcwd(), 'NVD_func'), join(os.getcwd(), 'NVD_diff'))
    # diffParser.parse('oldnvd', "../sundries/oldnvd", "../sundries/NVD_diff")
