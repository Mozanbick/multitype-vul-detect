import os
import pickle
import shutil
from slice2graph.gen_slice import *
from utils.objects.cpg import Cpg
from utils.objects import FPG
from configs import modelConfig as ModelConfig
from utils.embeddings import Corpus, generate_w2vModel, load_w2vModel
from utils.objects.dataset import GraphDataset


def program_slices_to_graphs(cpg: Cpg, points_file: str, label_path: str):
    fc_points, apu_points, ae_points, fp_points, fr_points = get_points_from_file(points_file=points_file)
    if label_path.endswith('.xml'):
        label_dict = get_vul_lines_from_xml(label_path)
    elif label_path.endswith('.pkl'):
        label_dict = get_vul_lines_from_pkl(label_path)
    elif label_path == "":
        label_dict = {}
    else:
        raise TypeError(f"Label file {label_path} must endswith `.xml` or `.pkl`")
    s1, v1, n1 = FC_slices_to_graphs(cpg, fc_points, label_dict)
    s2, v2, n2 = AUPU_slices_to_graphs(cpg, apu_points, label_dict)
    s3, v3, n3 = AE_slices_to_graphs(cpg, ae_points, label_dict)
    s4, v4, n4 = FP_slices_to_graphs(cpg, fp_points, label_dict)
    s5, v5, n5 = FR_slices_to_graphs(cpg, fr_points, label_dict)
    spg_list = s1 + s2 + s3 + s4 + s5
    save_path = join(ModelConfig.spgs_dir, f"spg_list_{ModelConfig.group}.pkl")
    if not exists(ModelConfig.spgs_dir):
        os.makedirs(ModelConfig.spgs_dir)
    with open(save_path, "wb") as fp:
        pickle.dump(spg_list, fp)
    time.sleep(0.5)
    print(f"vul slices number: {sum([v1, v2, v3, v4, v5])}")
    print(f"non-vul slices number: {sum([n1, n2, n3, n4, n5])}")
    return spg_list


def program_functions_to_graphs(cpg: Cpg):
    fpg_list = []
    vul_count = 0
    non_vul_count = 0
    for testID in cpg.methods:
        for method in cpg.methods[testID]:
            slice_list = list(method.node_id_set)
            items = method.filename.split("@@")
            label = int(items[1])
            assert label == 0 or label == 1
            if label == 1:
                vul_count += 1
            else:
                non_vul_count += 1
            fpg = FPG(testID, method, slice_list, label)
            if len(fpg.node_list) == 0:
                continue
            fpg_list.append(fpg)
    save_path = join(ModelConfig.fpgs_dir, f"fpg_list_{ModelConfig.group}.pkl")
    if not exists(ModelConfig.fpgs_dir):
        os.makedirs(ModelConfig.fpgs_dir)
    with open(save_path, "wb") as fp:
        pickle.dump(fpg_list, fp)
    time.sleep(0.5)
    print(f"vul functions number: {vul_count}")
    print(f"non-vul functions number: {non_vul_count}")
    return fpg_list


def program_slices_to_graphs_with_load():
    save_path = join(ModelConfig.spgs_dir, f"spg_list_{ModelConfig.group}.pkl")
    with open(save_path, "rb") as fp:
        spg_list = pickle.load(fp)
    return spg_list


def program_functions_to_graphs_with_load():
    save_path = join(ModelConfig.fpgs_dir, f"fpg_list_{ModelConfig.group}.pkl")
    with open(save_path, "rb") as fp:
        fpg_list = pickle.load(fp)
    return fpg_list


def program_slices_to_graphs_load_all():
    spg_list = []
    for file in os.listdir(ModelConfig.spgs_dir):
        path = os.path.join(ModelConfig.spgs_dir, file)
        with open(path, "rb") as fp:
            glist = pickle.load(fp)
        spg_list += glist
    return spg_list


def program_functions_to_graphs_load_all():
    fpg_list = []
    for file in os.listdir(ModelConfig.fpgs_dir):
        path = join(ModelConfig.fpgs_dir, file)
        with open(path, "rb") as fp:
            glist = pickle.load(fp)
        fpg_list += glist
    return fpg_list


def program_slices_to_graphs_load_test():
    spg_list = []
    for file in os.listdir(ModelConfig.spgs_dir):
        if 'test' in file:
            path = join(ModelConfig.spgs_dir, file)
            with open(path, "rb") as fp:
                glist = pickle.load(fp)
            spg_list += glist
    return spg_list


def graph_to_dataset(
        cpg: Cpg,
        points_file: str,
        label_path: str,
        corpus_path: str,
        w2v_path: str,
        save_path: str
):
    """
    ++ train embedding nn
    ++ embed nodes
    ++ convert spg into graph dataset
    """
    # spg_list = program_slices_to_graphs(cpg, points_file, label_path)
    spg_list = program_slices_to_graphs_with_load()
    # spg_list = program_functions_to_graphs_with_load()
    # spg_list = program_functions_to_graphs_load_all()
    # spg_list = program_slices_to_graphs_load_test()
    # generate corpus
    # corpus = Corpus(corpus_path)
    # for spg in tqdm(spg_list, desc="generating corpus"):
    #     corpus.add_corpus(spg.node_list)
    # corpus.save()
    # generate w2v_model
    # w2v = generate_w2vModel(corpus_path, w2v_path, size=ModelConfig.embed_dim)
    w2v = load_w2vModel(w2v_path)
    # embed and save
    mode = True if 'test' in ModelConfig.group else False
    dataset = GraphDataset(ModelConfig.dataset, save_path, test=mode)
    for spg in spg_list:
        spg.embed(ModelConfig.nodes_dim, w2v.wv)
        # print(spg)
        dataset.add_graph(spg)
    dataset.save()


def change_test_paths(dataset: str, save_path: str):
    """
    The original path of test dataset is `<save_path>/<dataset_name>/<vul_type>/dataset.bin`,
    we need to convert this path to `<save_path>/<vul_type>/<dataset_name>/dataset.bin`,
    for the convenience for the model test implementation.
    """
    # check sub folder structure
    if dataset not in os.listdir(save_path):
        return
    for vul_type in os.listdir(join(save_path, dataset)):
        for filename in os.listdir(join(save_path, dataset, vul_type)):
            src_path = join(save_path, dataset, vul_type, filename)
            dst_dir = join(save_path, vul_type, dataset)
            if not exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.move(src_path, join(dst_dir, filename))
    # delete original folder
    shutil.rmtree(join(save_path, dataset))


def graph_to_dataset_new(
        cpg: Cpg,
        points_file: str,
        label_path: str,
        corpus_path: str,
        w2v_path: str,
        save_path: str,
        args
):
    # generate slice/function program slice
    if not args.func_level:  # slice level
        if args.gen_graph:
            g_list = program_slices_to_graphs(cpg, points_file, label_path)
        else:
            try:
                g_list = program_slices_to_graphs_with_load()
            except FileNotFoundError:
                return
    else:  # function level
        if args.gen_graph:
            g_list = program_functions_to_graphs(cpg)
        else:
            try:
                g_list = program_functions_to_graphs_with_load()
            except FileNotFoundError:
                return
    # generate corpus
    if args.gen_corpus:
        corpus = Corpus(corpus_path)
        if not args.func_level:  # slice level
            all_list = program_slices_to_graphs_with_load()
        else:  # function level
            all_list = program_functions_to_graphs_with_load()
        for g in tqdm(all_list, desc="generating corpus..."):
            method = cpg.get_method_by_filename(g.testID, g.filenames.pop())
            corpus.add_corpus(g.node_list, method)
        corpus.save()
    # generate embedding model
    if args.gen_w2v:
        generate_w2vModel(corpus_path, w2v_path, size=ModelConfig.embed_dim)
    # convert slice/function program graph to model input dataset
    if args.g2dataset:
        # load embedding model
        w2v = load_w2vModel(w2v_path)
        # embed and save
        mode = True if 'test' in ModelConfig.group else False
        dataset = GraphDataset(ModelConfig.dataset, save_path, test=mode)
        for spg in tqdm(g_list, desc="convert graphs to dataset..."):
            method = cpg.get_method_by_filename(spg.testID, spg.filenames.pop())
            spg.embed(ModelConfig.nodes_dim, method, w2v.wv)
            dataset.add_graph(spg)
        dataset.save()
        if mode:
            change_test_paths(ModelConfig.dataset, save_path)
