import os
import time
import func_timeout
from slice2graph.slice_utils import *
from utils.objects import Cpg, Method
from typing import List
from tqdm import tqdm


def FC_slices(cpg: Cpg, point_list: List):
    """
    Generating FC type slices
    """
    fc_slices = {}
    for point_id in tqdm(point_list, desc="FC slices"):
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        try:
            fc_slices[point_id] = gen_slices(cpg, testID, method, point_id)
        except func_timeout.FunctionTimedOut:
            continue
    return fc_slices


def FC_slices_to_file(cpg: Cpg, point_list: List, save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = FC_slices(cpg, point_list)
    write_slices_to_file(slices, join(save_dir, "FC_slices.txt"))


def FC_slices_to_sequences(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]], save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = FC_slices(cpg, point_list)
    return gen_sequences(cpg, slices, label_dict, join(save_dir, "FC_slices.txt"))


def FC_slices_to_graphs(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]]):
    slices = FC_slices(cpg, point_list)
    return gen_graphs(cpg, slices, label_dict, "FC")


def AUPU_slices(cpg: Cpg, point_list: List):
    """
    Generating AU and PU slices
    """
    apu_slices = {}
    for point_id in tqdm(point_list, desc="AUPU slices"):
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        # point_id = method.get_parent_node(point_id)
        try:
            apu_slices[point_id] = gen_slices(cpg, testID, method, point_id)
        except func_timeout.FunctionTimedOut:
            continue
    return apu_slices


def AUPU_slices_to_file(cpg: Cpg, point_list: List, save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = AUPU_slices(cpg, point_list)
    write_slices_to_file(slices, join(save_dir, "AUPU_slices.txt"))


def AUPU_slices_to_sequences(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]], save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = AUPU_slices(cpg, point_list)
    return gen_sequences(cpg, slices, label_dict, join(save_dir, "AUPU_slices.txt"))


def AUPU_slices_to_graphs(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]]):
    slices = AUPU_slices(cpg, point_list)
    return gen_graphs(cpg, slices, label_dict, "AUPU")


def AE_slices(cpg: Cpg, point_list: List):
    """
    Generating AE slices
    """
    ae_slices = {}
    for point_id in tqdm(point_list, desc="AE slices"):
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        # point_id = method.get_parent_node(point_id)
        try:
            ae_slices[point_id] = gen_slices(cpg, testID, method, point_id)
        except func_timeout.FunctionTimedOut:
            continue
    return ae_slices


def AE_slices_to_file(cpg: Cpg, point_list: List, save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = AE_slices(cpg, point_list)
    write_slices_to_file(slices, join(save_dir, "AE_slices.txt"))


def AE_slices_to_sequences(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]], save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = AE_slices(cpg, point_list)
    return gen_sequences(cpg, slices, label_dict, join(save_dir, "AE_slices.txt"))


def AE_slices_to_graphs(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]]):
    slices = AE_slices(cpg, point_list)
    return gen_graphs(cpg, slices, label_dict, "AE")


def FP_slices(cpg: Cpg, point_list: List):
    """
    Generating FP slices
    """
    fp_slices = {}
    for point_id in tqdm(point_list, desc="FP slices"):
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        try:
            fp_slices[point_id] = gen_slices(cpg, testID, method, point_id)
        except func_timeout.FunctionTimedOut:
            continue
    return fp_slices


def FP_slices_to_file(cpg: Cpg, point_list: List, save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = FP_slices(cpg, point_list)
    write_slices_to_file(slices, join(save_dir, "FP_slices.txt"))


def FP_slices_to_sequences(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]], save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = FP_slices(cpg, point_list)
    return gen_sequences(cpg, slices, label_dict, join(save_dir, "FP_slices.txt"))


def FP_slices_to_graphs(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]]):
    slices = FP_slices(cpg, point_list)
    return gen_graphs(cpg, slices, label_dict, "FP")


def FR_slices(cpg: Cpg, point_list: List):
    """
    Generating FR slices
    """
    fr_slices = {}
    for point_id in tqdm(point_list, desc="FR slices"):
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        try:
            fr_slices[point_id] = gen_slices(cpg, testID, method, point_id)
        except func_timeout.FunctionTimedOut:
            continue
    return fr_slices


def FR_slices_to_file(cpg: Cpg, point_list: List, save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = FR_slices(cpg, point_list)
    write_slices_to_file(slices, join(save_dir, "FR_slices.txt"))


def FR_slices_to_sequences(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]], save_dir: str):
    if not exists(save_dir):
        os.makedirs(save_dir)
    slices = FR_slices(cpg, point_list)
    return gen_sequences(cpg, slices, label_dict, join(save_dir, "FR_slices.txt"))


def FR_slices_to_graphs(cpg: Cpg, point_list: List, label_dict: Dict[str, Dict[str, List]]):
    slices = FR_slices(cpg, point_list)
    return gen_graphs(cpg, slices, label_dict, "FR")


def program_slices(cpg: Cpg, points_file: str, label_path: str):
    """
    Get node id of candidate vulnerable points from points file
    Generate 6 kinds of slices and write them into files
    """
    fc_points, apu_points, ae_points, fp_points, fr_points = get_points_from_file(points_file=points_file)
    label_dict = get_vul_lines_from_xml(label_path)


def program_slices_to_files(cpg: Cpg, points_file: str, save_dir: str):
    fc_points, apu_points, ae_points, fp_points, fr_points = get_points_from_file(points_file=points_file)
    FC_slices_to_file(cpg, fc_points, save_dir)
    AUPU_slices_to_file(cpg, apu_points, save_dir)
    AE_slices_to_file(cpg, ae_points, save_dir)
    FP_slices_to_file(cpg, fp_points, save_dir)
    FR_slices_to_file(cpg, fr_points, save_dir)


def program_slices_to_sequences(cpg: Cpg, points_file: str, label_path: str, save_dir: str):
    fc_points, apu_points, ae_points, fp_points, fr_points = get_points_from_file(points_file=points_file)
    if label_path.endswith('.xml'):
        label_dict = get_vul_lines_from_xml(label_path)
    else:
        label_dict = get_vul_lines_from_pkl(label_path)
    v1, n1 = FC_slices_to_sequences(cpg, fc_points, label_dict, save_dir)
    v2, n2 = AUPU_slices_to_sequences(cpg, apu_points, label_dict, save_dir)
    v3, n3 = AE_slices_to_sequences(cpg, ae_points, label_dict, save_dir)
    v4, n4 = FP_slices_to_sequences(cpg, fp_points, label_dict, save_dir)
    v5, n5 = FR_slices_to_sequences(cpg, fr_points, label_dict, save_dir)
    time.sleep(0.5)
    print(f"vul slices number: {sum([v1, v2, v3, v4, v5])}")
    print(f"non-vul slices number: {sum([n1, n2, n3, n4, n5])}")
