import os
import shutil
from os.path import join, isdir, exists
from os import listdir


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
        if 'dtest' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'dtest'))
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
            types = 'dtest' if test else 'train'
            if test:
                groups = 'group' + str(self.test_group)
            else:
                groups = 'group' + str(self.train_group)
            save_dir = join(self.save_dir, types, groups)
            if groups not in listdir(join(self.save_dir, types)):
                os.makedirs(save_dir)
            shutil.copyfile(p, join(save_dir, save_name))

    def collect_with_split_fan(self, path, test=False, max_testcase=200):
        test_set = [
            'DoS', 'Exec Code', 'Overflow', 'XSS', 'Dir. Trav', 'Bypass',
            '+Info', '+Priv', 'Men. Corr', 'Sql'
        ]
        items = listdir(path)
        if 'train' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'train'))
        if 'dtest' not in listdir(self.save_dir):
            os.makedirs(join(self.save_dir, 'dtest'))
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
            types = 'dtest' if test else 'train'
            if test:
                groups = 'group' + str(self.test_group)
            else:
                groups = 'group' + str(self.train_group)
            save_dir = join(self.save_dir, types, groups)
            if groups not in listdir(join(self.save_dir, types)):
                os.makedirs(save_dir)
            shutil.copyfile(p, join(save_dir, save_name))

    def fan_rearrange(self, path: str):
        items = listdir(path)
        for item in items:
            p = join(path, item)
            if isdir(p):
                self.fan_rearrange(p)
                if item.isdigit():
                    shutil.rmtree(p)
                continue
            idx = path.split(os.sep)[-1]
            if item.endswith('.c'):
                filename = idx + '_' + item
                shutil.copyfile(p, join(path, '../', filename))
            elif item.endswith('.diff'):
                if not os.path.exists(join(path, '../', item)):
                    shutil.copyfile(p, join(path, '../', item))

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


if __name__ == '__main__':
    to_dir = "/data/zjh/SG/joern/data/cfan/"
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    # collectCFiles = CollectCFiles(os.getcwd(), to_dir)
    # collectCFiles.fan_rearrange(os.getcwd())
    # collectCFiles.collect_with_split_fan(collectCFiles.base_dir, max_testcase=200)
    # collectCFiles = CollectCFiles(join(os.getcwd(), 'oldnvd'), '')
    # collectCFiles.collect_old_nvd(collectCFiles.base_dir)
    collectCFiles = CollectCFiles(join(os.getcwd(), 'NVD_func'), join(os.getcwd(), 'nnvd'))
    collectCFiles.collect_new_nvd(collectCFiles.base_dir)
