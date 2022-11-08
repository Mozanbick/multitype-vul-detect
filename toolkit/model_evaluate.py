"""
version: 2022/11/7
author: zjh

This script is going to draw pictures of learning loss, accuracy etc.
"""
import os
import re
import matplotlib


def draw_loss(list_loss: list):
    pass


def log_parser(log_path: str):
    with open(log_path, "r", encoding="utf-8") as fp:
        contents = fp.read()
    l_contents = contents.split('\nEPOCH ###### ')
    l_loss_train = []
    l_loss_val = []
    l_acc = []
    l_f1 = []
    l_pre = []
    l_re = []
    best_epoch = set()
    # patterns
    r_loss_train = r"loss \d+\.\d+ with epoch time \d+\.\d+ s & computation time \d+\.\d+ s"
    r_loss_val = r"loss \d+\.\d+ in validation"
    r_result = r"Validation : Accuracy \d+\.\d+% | F1 score \d+\.\d+% | Precision \d+\.\d+% | Recall \d+\.\d+%"
    r_epoch = r"best epoch is EPOCH \d+, val_acc is \d+\.\d+%"
    for item in l_contents:
        item = item.strip()
        # search
        s_loss_train = re.search(r_loss_train, item).group()
        s_loss_val = re.search(r_loss_val, item).group()
        s_results = re.search(r_result, item).group()
        s_epoch = re.search(r_epoch, item).group()
        # extract
        loss_train = re.findall(r"\d+\.\d+", s_loss_train)
        l_loss_train.append(float(loss_train[0]))
        loss_val = re.findall(r"\d+\.\d+", s_loss_val)
        l_loss_val.append(float(loss_val[0]))
        results = re.findall(r"\d+\.\d+", s_results)
        l_acc.append(float(results[0]))
        l_f1.append(float(results[1]))
        l_pre.append(float(results[2]))
        l_re.append(float(results[3]))
        epoch = re.findall(r"\d+", s_epoch)
        best_epoch.add(int(epoch[0]))

    return l_loss_train, l_loss_val, l_acc, l_f1, l_pre, l_re, max(best_epoch)