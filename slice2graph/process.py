import os
import time
from slice2graph.gen_slice import program_slices_to_files, program_slices_to_sequences
from utils.objects import Cpg
from os.path import exists


if __name__ == '__main__':
    saveDir = "../joern/data/sequences/"
    points_file = "../joern/joern-cli/results/AllVulPoints.txt"
    cpg_path = "../joern/joern-cli/results/"
    xml_path = "../joern/data/SARD_testcaseinfo.xml"
    start_time = time.time()
    cpg = Cpg(cpg_path)
    program_slices_to_sequences(cpg, points_file, xml_path, saveDir)
    end_time = time.time()
    print(f"Total process time: {end_time - start_time} s.")
