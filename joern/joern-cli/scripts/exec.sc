@main def main(cpg_path: String, save_path: String, funcs_path: String) = {
    // Args:
    // cpg_path: Path to load cpg file
    // save_path: Path to save funcs info and focus points files
    // funcs_path: Path to load sensitive_funcs.txt
    importCpg(cpg_path)
    extractFuncs(save_path)
    getPoints(save_path, funcs_path)
}