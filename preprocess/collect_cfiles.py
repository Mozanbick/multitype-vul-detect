import os
import shutil
import sys
import hashlib
from os.path import join, isdir, exists
from os import listdir


def file_hash(path: str):
    with open(path, "r") as fp:
        return hashlib.md5(fp.read().encode()).hexdigest()


class CollectCFiles:

    def __init__(self, base: str, save: str):
        self.base_dir = base
        self.save_dir = save
        self.train_count = 0
        self.test_count = 0
        self.train_group = 0
        self.test_group = 0

    def collect(self, path, max_testcase=500):
        items = listdir(path)
        for item in items:
            p = join(path, item)
            if isdir(p):
                self.collect(path)
            if not item.endswith('.c'):
                continue
            title = p.removeprefix(self.base_dir)
            names = title.split(os.sep)
            prefix = ''
            for _ in range(len(names)-1):
                prefix += names[_] + '#'
            filename = names[-1]
            label = 1 if 'OLD' in filename else 0
            save_name = prefix + '@@' + str(label) + '@@' + filename
            if prefix not in listdir(self.save_dir):
                os.makedirs(join(self.save_dir, prefix))
            shutil.copyfile(p, join(self.save_dir, prefix, save_name))

    def collect_with_split_nvd(self, path, test=False, max_testcase=500):
        test_set = [
            'DoS', 'Exec Code', 'Overflow', 'XSS', 'Dir. Trav', 'Bypass',
            '+Info', '+Priv', 'Men. Corr', 'Sql'
        ]
        items = listdir(path)
        if 'train' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'train'))
        if 'test' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'test'))
        for item in items:
            p = join(path, item)
            sub_test = test
            if isdir(p):
                if item in test_set:
                    sub_test = True
                if item.startswith('CVE'):
                    if test:
                        self.test_count += 1
                    else:
                        self.train_count += 1
                    if self.test_count >= max_testcase:
                        self.test_group += 1
                        self.test_count = 0
                    if self.train_count >= max_testcase:
                        self.train_group += 1
                        self.train_count = 0
                self.collect_with_split_nvd(p, sub_test, max_testcase)
            if not item.endswith('.c'):
                continue
            title = p.removeprefix(self.base_dir)
            names = title.split(os.sep)
            prefix = ''
            for _ in range(len(names) - 1):
                prefix += names[_] + '#'
            filename = names[-1]
            label = 1 if 'OLD' in filename else 0
            save_name = prefix + '@@' + str(label) + '@@' + filename
            types = 'test' if test else 'train'
            if test:
                groups = 'group' + str(self.test_group)
            else:
                groups = 'group' + str(self.train_group)
            save_dir = join(self.save_dir, types, groups)
            if groups not in listdir(join(self.save_dir, types)):
                os.makedirs(save_dir)
            shutil.copyfile(p, join(save_dir, save_name))

    def collect_with_split_fan(self, path, test=False, max_testcase=300):
        test_set = [
            'DoS', 'Exec Code', 'Overflow', 'XSS', 'Dir. Trav', 'Bypass',
            '+Info', '+Priv', 'Men. Corr', 'Sql'
        ]
        items = listdir(path)
        if 'train' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'train'))
        if 'test' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'test'))
        for item in items:
            p = join(path, item)
            sub_test = test
            if isdir(p):
                if item in test_set:
                    sub_test = True
                if item.startswith('CVE'):
                    if test:
                        self.test_count += 1
                    else:
                        self.train_count += 1
                    if self.test_count >= max_testcase:
                        self.test_group += 1
                        self.test_count = 0
                    if self.train_count >= max_testcase:
                        self.train_group += 1
                        self.train_count = 0
                self.collect_with_split_fan(p, sub_test, max_testcase)
            if not item.endswith('.c'):
                continue
            title = p.removeprefix(self.base_dir)
            names = title.split(os.sep)
            prefix = ''
            for _ in range(len(names) - 1):
                prefix += names[_] + '#'
            filename = names[-1]
            label = 1 if 'before' in filename else 0
            save_name = prefix + '@@' + str(label) + '@@' + filename
            types = 'test' if test else 'train'
            if test:
                groups = 'group' + str(self.test_group)
            else:
                groups = 'group' + str(self.train_group)
            save_dir = join(self.save_dir, types, groups)
            if groups not in listdir(join(self.save_dir, types)):
                os.makedirs(save_dir)
            shutil.copyfile(p, join(save_dir, save_name))

    def fan_rearrange(self, path: str, hash_list=None):
        items = listdir(path)
        for item in items:
            if item.startswith('CVE'):
                hash_list = []
            p = join(path, item)
            if isdir(p):
                self.fan_rearrange(p, hash_list)
                if item.isdigit():
                    shutil.rmtree(p)
                continue
            idx = path.split(os.sep)[-1]
            if item.endswith('.c'):
                filename = idx + '_' + item
                shutil.copyfile(p, join(path, '../', filename))
            elif item.endswith('.diff'):
                md5 = file_hash(p)
                if md5 not in hash_list:
                    hash_list.append(md5)
                    shutil.copyfile(p, join(path, '../', idx + '_' + item))

    def collect_old_nvd(self, path, max_testcase=500):
        for item in listdir(path):
            p = join(path, item)
            if isdir(p):
                self.collect_old_nvd(p, max_testcase)
            if not p.endswith('.c'):
                continue
            if '_VULN_' in item:
                label = 1
            else:
                label = 0
            if label == 1:
                testID = item.split('_VULN_')[0].replace('_', '-')
                name = item.split('_VULN_')[1]
            else:
                testID = item.split('_PATCHED_')[0].replace('_', '-')
                name = item.split('_PATCHED_')[1]
            filename = testID + '@@' + str(label) + '@@' + name
            os.rename(p, join(path, filename))

    def collect_new_nvd(self, path, max_testcase=500):
        for item in listdir(path):
            p = join(path, item)
            if isdir(p):
                if item.startswith('CVE'):
                    self.train_count += 1
                if self.train_count >= max_testcase:
                    self.train_group += 1
                    self.train_count = 0
                self.collect_new_nvd(p, max_testcase)
            if not p.endswith('.c'):
                continue
            names = p.split(os.sep)
            cve = [i for i, s in enumerate(names) if s.startswith('CVE')]  # one item
            testID = names[cve[0] - 1] + '#' + names[cve[0]]
            filename = names[-1]
            label = 1 if 'OLD' in filename else 0
            save_name = testID + '@@' + str(label) + '@@' + filename
            groups = 'group' + str(self.train_group)
            save_dir = join(self.save_dir, groups)
            if not exists(save_dir):
                os.makedirs(save_dir)
            shutil.copyfile(p, join(save_dir, save_name))


def func_wrapper(base: str, save: str):
    collectCFiles = CollectCFiles(base, save)
    if 'fan' in base.lower():
        collectCFiles.fan_rearrange(collectCFiles.base_dir)
        collectCFiles.collect_with_split_fan(collectCFiles.base_dir)
    elif 'convertedNVD' in base:
        collectCFiles.collect_with_split_nvd(collectCFiles.base_dir)
    elif 'newnvd' in base.lower():
        collectCFiles.collect_new_nvd(collectCFiles.base_dir)
    elif 'oldnvd' in base.lower():
        collectCFiles.collect_old_nvd(collectCFiles.base_dir)


if __name__ == '__main__':
    # need for 2 args: source dataset dir and save dir
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]
    assert os.path.exists(from_dir)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    func_wrapper(from_dir, to_dir)
