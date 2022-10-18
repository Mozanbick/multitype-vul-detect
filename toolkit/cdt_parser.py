import os


def parse(path: str):
    with open(path, "r") as fp:
        contents = fp.readlines()
    label_attr = False
    label_ast = False
    num = []
    ast = []
    for line in contents:
        if "-----joern-----" in line:
            break
        if label_ast:
            a = line.strip()
            ast.append(a)
        if "-----ast_node-----" in line:
            label_attr = False
            label_ast = True
        if label_attr:
            num = line.strip().split(';')
            for x in num:
                if x == "" or x == '\n':
                    num.remove(x)
        if "-----attribute-----" in line:
            label_attr = True
    return num, ast


def query(attribute: str):
    idx = [i for i, x in enumerate(attr_list) if x == attribute]
    if len(idx) == 0:
        print("None")
    for p in idx:
        print(p, ",  ", ast_list[p])


def check(idx: int):
    print(attr_list[idx])


if __name__ == '__main__':
    file_path = "C:\\Users\\ZZH\\Desktop\\9b6b7d5c-138a-4cda-9b15-4397358155e9.c-r_core_cmd_subst_i.txt"
    attr_list, ast_list = parse(file_path)
    print("OK.")
    while True:
        f, attr = input().strip().split(' ')
        if f == "query":
            query(attr)
        elif f == "check":
            check(int(attr))
