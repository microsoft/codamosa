"""
Runs edit distance calculations on all the generated data for a particular configuration.

The data created is a map from project to a tuple with 3 elements:
- list of maximum simlarities for all Codex-generated testcases for that project
- list of lengths (in characters) for all Codex-generated testcases for that project
- list of pairs of (Codex-generated test case body, extracted test case body with
   maximum similarity) for all Codex-generated testcases for that project. 
"""
import ast
import os
import sys
from typing import Dict, List

import astor
import editdistance
from tqdm import tqdm

CACHED_TEST_BODIES: Dict[str, List[str]] = {}

from itertools import cycle

import matplotlib.pyplot as plt

lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)


def get_body_as_str(node: ast.FunctionDef) -> str:
    fn_as_string = "".join([astor.to_source(elem) for elem in node.body])
    return fn_as_string


class TestBodyCollector(ast.NodeVisitor):
    def __init__(self):
        self.fn_bodies: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for elem in node.body:
            self.visit(elem)
        if "test" in node.name.lower():
            self.fn_bodies.append(get_body_as_str(node))

    def get_bodies(self) -> List[str]:
        return self.fn_bodies


def get_project_tests(source_code_dir: str) -> List[str]:
    if source_code_dir not in CACHED_TEST_BODIES:
        reference_test_cases: List[str] = []
        print("Collecting test cases for", source_code_dir)
        for base_path, _, file_names in tqdm(os.walk(source_code_dir)):
            for file_name in file_names:
                if file_name.endswith(".py"):
                    try:
                        file_contents = open(os.path.join(base_path, file_name)).read()
                        module = ast.parse(file_contents)
                        collector = TestBodyCollector()
                        collector.visit(module)
                        reference_test_cases.extend(collector.get_bodies())
                    except SyntaxError:
                        pass

        CACHED_TEST_BODIES[source_code_dir] = reference_test_cases
    return CACHED_TEST_BODIES[source_code_dir]


def main(benchmark_name: str, source_code_dir: str, results_base: str):
    benchmark_name_split = benchmark_name.split(".")
    benchmark_path = os.path.join(*benchmark_name_split[:-1])
    benchmark_file = benchmark_name_split[-1] + ".py"

    # Collect all the source code files in source_code_dir other than the benchmark name.
    # Get all the test bodies defined in tehm
    reference_tests: List[str] = get_project_tests(source_code_dir)

    # Take all the test cases from all the generations
    test_cases = []
    for i in range(16):
        generations_file = os.path.join(
            results_base, benchmark_name + f"-{i}", "codex_generations.py"
        )
        if os.path.isfile(generations_file):
            with open(generations_file) as f:
                cur_gen = []
                for line in f:
                    if line.startswith("# Generated at"):
                        test_cases.append("".join(cur_gen))
                        cur_gen = []
                    else:
                        cur_gen.append(line)
                if len(cur_gen) > 0:
                    test_cases.append("".join(cur_gen))

    # Collect each test case.
    # Compare to all the harvested testcsaes.
    max_sim_list = []
    len_list = []
    test_case_pairs = []
    for test_case in test_cases:
        try:
            test_case_module = ast.parse(test_case)
        except SyntaxError:
            print("this test case didn't parse")
            print(test_case)
            continue
        if len(test_case_module.body) == 0:
            continue
        elif not isinstance(test_case_module.body[0], ast.FunctionDef):
            print("This test case is weird:")
            print(test_case)
            continue
        test_case_node = test_case_module.body[0]
        normalized_test_case_body = get_body_as_str(test_case_node)
        max_sim = 0
        samest_test_case = "<NONE>"
        for reference_test in reference_tests:
            distance = editdistance.distance(normalized_test_case_body, reference_test)
            max_len = max(len(normalized_test_case_body), len(reference_test))
            norm_distance = distance / max_len
            test_sim = 1 - norm_distance
            if test_sim > max_sim:
                max_sim = test_sim
                samest_test_case = reference_test
        if max_sim >= 1.2:
            print(
                f"============================\n{normalized_test_case_body}\n    >>>> IS MOST SIMILAR TO (sim = {max_sim}) <<<<\n{samest_test_case}"
            )
        if max_sim == 1:
            print(
                f"=======EXACTLY COPIED TEST CASE=======\n{normalized_test_case_body}"
            )
        max_sim_list.append(max_sim)
        len_list.append(len(normalized_test_case_body))
        test_case_pairs.append((normalized_test_case_body, samest_test_case))

    return max_sim_list, len_list, test_case_pairs


def plot_cumulative_similarities(max_sim_list: List[float], proj_name=""):
    cumulative_sims = []
    sims_under_current = 0
    last_sim = 0
    for max_sim in max_sim_list:
        if max_sim > last_sim:
            cumulative_sims.append((last_sim, sims_under_current))
            last_sim = max_sim
        sims_under_current += 1
    if proj_name == "":
        plt.plot(
            [cs[0] for cs in cumulative_sims],
            [cs[1] / sims_under_current for cs in cumulative_sims],
        )
    else:
        plt.plot(
            [cs[0] for cs in cumulative_sims],
            [cs[1] / sims_under_current for cs in cumulative_sims],
            label=proj_name,
            linestyle=next(linecycler),
        )
    bot, top = plt.ylim()
    plt.ylim((0, top))
    ax = plt.gca()

    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if len(sys.argv) != 5:
        print(
            f"usage: python3 {sys.argv[0]} test_apps_base results_base csv_with_all_modules output_file"
        )
        exit(1)

    import csv
    import pickle
    from collections import defaultdict

    test_apps_base = sys.argv[1]
    results_base = sys.argv[2]

    proj_to_benchs = defaultdict(list)
    to_pickle = {}
    all_data_to_pickle = {}
    with open(sys.argv[3]) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            project_path_raw = row[0]
            project_path_split = project_path_raw.split("/")
            final_project_path = os.path.join(
                test_apps_base, project_path_split[0], project_path_split[1]
            )
            benchmark_name = row[1]
            assert benchmark_name[-1] != "\n"
            proj_to_benchs[final_project_path].append(benchmark_name)
    for proj_path, benchs in proj_to_benchs.items():
        all_similarities = []
        all_sim_len = 0
        all_lens = []
        all_pairs = []
        print(proj_path.split("/")[-1], " processing...")
        for bench in tqdm(benchs):
            ret_sims, ret_lens, pairs = main(bench, proj_path, results_base)
            all_similarities.extend(ret_sims)
            all_lens.extend(ret_lens)
            all_pairs.extend(pairs)
            all_sim_len += len(ret_sims)
        assert len(all_similarities) == all_sim_len
        sorted_similarities = sorted(all_similarities)
        print(all_sim_len)
        proj_name = proj_path.split("/")[-1]
        plot_cumulative_similarities(sorted_similarities, proj_name)
        all_data_to_pickle[proj_name] = (all_similarities, all_lens, all_pairs)

    output_file = sys.argv[4]
    pickle_file = open(output_file, "wb")
    pickle.dump(all_data_to_pickle, pickle_file)
    pickle_file.close()
