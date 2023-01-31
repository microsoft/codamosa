"""
If used with a directory name as argument, gathers all the statistics.csv files in the specified directory into a sensible structure.
Assume the following directory structure:
- {input_dir}
   - {config_i}
        - {module_j}-{iteration}
            - statistics.csv
            - <passing_tests.py>
            - <failing_tests.py>
            - codamosa_time.csv (optional)

Contains various functions for plotting and exploratory data analysis. 
"""
import collections
import os
import pickle
import random
import sys
from enum import Enum
from typing import Any, List, Tuple

import matplotlib
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats
import tqdm
from matplotlib.lines import Line2D

MAX_TIME = 600
# Minimum number of reps to include for analysis.
MIN_REPS = 10
PLOTS_DIR = None


def get_nice_name(config_name: str):
    if config_name == "codamosa-0.8-uninterp":
        return "CodaMOSA"
    elif config_name == "mosa":
        return "MOSA"
    elif config_name == "codex-only":
        return "CodexOnly"
    elif config_name == "codamosa-0.8":
        return "CodaMOSA-No-Uninterp"
    elif config_name == "codamosa-0.2-uninterp":
        return "CodaMOSA-Temp-0.2"
    elif config_name == "codamosa-0.8-uninterp-no-targeting":
        return "CodaMOSA-Random-Target"
    elif config_name == "codamosa-0.8-uninterp-small":
        return "CodaMOSA-TestCase-Prompt"
    else:
        return config_name


def not_sig_color(opacity):
    return (0.8, 0.40, 0, opacity)


def get_color(config_name: str, opacity: float):
    if config_name == "codamosa-0.8-uninterp":
        return (0, 0.45, 0.70, opacity)
    else:
        return (0, 0, 0, opacity)
    # elif config_name == 'mosa':
    #     return (0.80, 0.40, 0, opacity)
    # elif config_name == 'codex-only':
    #     return (0.9, 0.6, 0, opacity)
    # elif config_name == 'codamosa-0.8':
    #     return (0.35, 0.70, 0.90, opacity)
    # elif config_name == 'codamosa-0.2-uninterp':
    #     return (0, 0, 0, opacity)
    # elif config_name == 'codamosa-0.8-uninterp-no-targeting':
    #     return (0, 0.6, 0.5, opacity)
    # elif config_name == 'codamosa-0.8-uninterp-small':
    #     return (0.8, 0.6, 0.7, opacity)
    # else: assert False


def get_marker(config_name):
    if config_name == "codamosa-0.8-uninterp":
        return "+"
    else:
        return "x"
    # elif config_name == 'mosa':
    #     return 'x'
    # elif config_name == 'codex-only':
    #     return 'x'
    # elif config_name == 'codamosa-0.8':
    #     return '1'
    # elif config_name == 'codamosa-0.2-uninterp':
    #     return 'd'
    # elif config_name == 'codamosa-0.8-uninterp-no-targeting':
    #     return '*'
    # elif config_name == 'codamosa-0.8-uninterp-small':
    #     return '2'
    # else: assert False


generation_stat_names = [
    "ParsableStatements",
    "ParsedStatements",
    "UninterpStatements",
    "LLMStageSavedTests",
    "LLMNeededUninterpreted",
    "LLMNeededExpansion",
    "LLMCalls",
    "LLMQueryTime",
]
GenerationInfo = collections.namedtuple(
    "GenerationInfo",
    [
        "parsable_stmts",
        "parsed_stmts",
        "uninterp_stmts",
        "saved_tests",
        "saved_uninterp_tests",
        "saved_expansion_tests",
        "llm_calls",
        "llm_time",
    ],
)


class CoverageContainer:
    def __init__(self):
        """
        Storage is indexed first by config name, then by module name.
        """
        self._storage = {}
        self._other_stats_storage = {}
        self._generations_storage = {}

    # Getters

    def get_configs(self):
        """Gets the list of configurations."""
        return set(self._storage.keys())

    def get_modules(self):
        """Gets the set of modules."""
        modules = set()
        for config in self._storage:
            modules.update(self._storage[config].keys())
        return modules

    def get_common_modules(self, config_lst: List[str]):
        """Get modules on which all configs in config_lst have enough repetitions"""
        all_modules = self.get_modules()
        discarded_modules = collections.defaultdict(list)
        for config in config_lst:
            all_obs = cc._storage[config]
            for module in all_modules:
                if module not in all_obs:
                    discarded_modules[module].append((config, 0))
                else:
                    num_obs = len(all_obs[module])
                    if num_obs < MIN_REPS:
                        discarded_modules[module].append((config, num_obs))
        for module, configs_obs_lst in discarded_modules.items():
            print(f"Removing {module} from analysis")
            for config_obs in configs_obs_lst:
                config, num_obs = config_obs
                print(f"    ==> {config} had {num_obs} observations")
        return all_modules.difference(discarded_modules.keys())

    def get_values_of_stat(self, stat_name, module_name, config_name):
        return self._other_stats_storage[config_name][module_name][stat_name]

    def get_observations(self, module_name: str, config_name: str):
        """Gets the coverage data for the module with `module_name` and configuration `config_name`."""
        if config_name in self._storage:
            if module_name in self._storage[config_name]:
                return self._storage[config_name][module_name]
        return []

    def get_codamosa_generations(self, module_name: str, config_name: str):
        """Gets the codamosa generation data for the module with `module_name` and configuration `config_name`."""
        if config_name in self._generations_storage:
            if module_name in self._generations_storage[config_name]:
                return self._generations_storage[config_name][module_name]
        return []

    def get_generation_statistics(self, module_name: str, config_name: str):
        """
        Get the generation statistics for the given module name and config_name
        """

        def get_stat(stat_name):
            """Fixup any statistics that have no observations and put them in an array"""
            all_values = cc.get_values_of_stat(stat_name, module_name, config_name)
            all_values = [0 if v == "" else v for v in all_values]
            return np.array(all_values)

        stat_values = []
        for stat_name in generation_stat_names:
            stat_values.append(get_stat(stat_name))

        return GenerationInfo(*stat_values)

    # Setters

    def add_other_stat_for_module(
        self, module_name: str, config_name: str, stat_name: str, stat_data: Any
    ):
        try:
            float_data = float(stat_data)
            int_data = int(float_data)
            if int_data == float_data:
                stat_data = int_data
            else:
                stat_data = float_data
        except ValueError:
            pass

        if config_name in self._other_stats_storage:
            if module_name in self._other_stats_storage[config_name]:
                if stat_name in self._other_stats_storage[config_name][module_name]:
                    self._other_stats_storage[config_name][module_name][
                        stat_name
                    ].append(stat_data)
                else:
                    self._other_stats_storage[config_name][module_name][stat_name] = [
                        stat_data
                    ]
            else:
                self._other_stats_storage[config_name][module_name] = {
                    stat_name: [stat_data]
                }
        else:
            self._other_stats_storage[config_name] = {
                module_name: {stat_name: [stat_data]}
            }

    def add_observation_for_module(
        self, module_name: str, config_name: str, coverage_data: np.array
    ):
        """
        Adds a new coverage data observation for the module with `module_name`, when run with configuration
         `config_name`.
        """
        if config_name in self._storage:
            if module_name in self._storage[config_name]:
                self._storage[config_name][module_name].append(coverage_data)
            else:
                self._storage[config_name][module_name] = [coverage_data]
        else:
            self._storage[config_name] = {module_name: [coverage_data]}

    def add_codamosa_generations_for_module(
        self, module_name: str, config_name: str, obs_data: np.array
    ):
        """
        Adds a new codamosa generation observation for the module with `module_name`, when run with configuration
         `config_name`.
        """
        if config_name in self._generations_storage:
            if module_name in self._generations_storage[config_name]:
                self._generations_storage[config_name][module_name].append(obs_data)
            else:
                self._generations_storage[config_name][module_name] = [obs_data]
        else:
            self._generations_storage[config_name] = {module_name: [obs_data]}

    # Statistical testing helpers

    def test_for_significant_final_cov(self, module_name: str, configs=None):
        """
        Conducts a Mann-Whitney U-Test to determine whether the final coverage achieved is
        significantly greater for any pair of configurations, for the module `module_name`.
        """
        if configs is None:
            configs = self.get_configs()
        output = {}
        for config1 in configs:
            for config2 in configs:
                if config1 != config2:
                    observations1 = [
                        o[-1] for o in self.get_observations(module_name, config1)
                    ]
                    observations2 = [
                        o[-1] for o in self.get_observations(module_name, config2)
                    ]
                    if len(observations1) > 0 and len(observations2) > 0:
                        u, p = scipy.stats.mannwhitneyu(
                            observations1, observations2, alternative="greater"
                        )
                        if p < 0.05:
                            output[(config1, config2)] = (u, p)
        return output

    def test_for_significant(self, config_to_data):
        """
        Conducts a Mann-Whitney U-Test to determine whether the data in config_to_data is
        significantly greater for any pair of configurations.
        """
        output = {}
        for config1 in config_to_data.keys():
            for config2 in config_to_data.keys():
                if config1 != config2:
                    observations1 = config_to_data[config1]
                    observations2 = config_to_data[config2]
                    if len(observations1) > 0 and len(observations2) > 0:
                        u, p = scipy.stats.mannwhitneyu(
                            observations1, observations2, alternative="greater"
                        )
                        if p < 0.05:
                            output[(config1, config2)] = (u, p)
        return output

    def ttest_for_significant_final_cov(self, module_name: str):
        """
        Conducts a Whelch's Test to determine whether the final coverage achieved is
        significantly greater for any pair of configurations, for the module `module_name`.
        """
        configs = self.get_configs()
        output = {}
        for config1 in configs:
            for config2 in configs:
                if config1 != config2:
                    observations1 = [
                        o[-1] for o in self.get_observations(module_name, config1)
                    ]
                    observations2 = [
                        o[-1] for o in self.get_observations(module_name, config2)
                    ]
                    if len(observations1) > 0 and len(observations2) > 0:
                        u, p = scipy.stats.ttest_ind(
                            observations1,
                            observations2,
                            equal_var=False,
                            alternative="greater",
                        )
                        if p < 0.05:
                            print(
                                f"[{module_name}] Found {config1} had signficiantly higher coverage than {config2} (p={p})"
                            )
                            output[(config1, config2)] = (u, p)
        return output

    # Exploratory Data Analysis

    def compare_with_scaled_query_time(self, module_name: str, scale: float):
        """
        Compares the performance of CODAMOSA to MOSA if CodaMOSA's query time is reduced.
        Scale: how much to scalet the query time
        """
        codamosa_obs = self.get_observations(module_name, "codamosa-0.8-uninterp")
        codamosa_final_values = [o[-1] for o in codamosa_obs]
        mosa_obs = self.get_observations(module_name, "mosa")
        query_time = self.get_values_of_stat(
            "LLMQueryTime", module_name, "codamosa-0.8-uninterp"
        )
        query_time = [0 if qt == "" else qt for qt in query_time]
        avg_query_time = np.mean(query_time)
        scaled_time = avg_query_time * scale
        total_time = round(600 - avg_query_time + scaled_time)
        mosa_final_values = [o[total_time] for o in mosa_obs]

        print(f"Average query time: {avg_query_time}")
        print(f"Reduced query time: {scaled_time}, gives search time: {total_time}")
        print(f"Codamosa avg cov at end: {np.mean(codamosa_final_values)}")
        print(f"Mosa avg cov at time {total_time}: {np.mean(mosa_final_values)}")

    def get_average_final_coverage_by_config(self, module_name: str):
        """Returns a map of the average end coverage for each configuration, for the
        given module `module_name`
        """
        result = {}
        for config in self.get_configs():
            final_observations = [
                o[-1] for o in self.get_observations(module_name, config)
            ]
            result[config] = np.mean(final_observations)
        colors = [
            "#4477AA",
            "#66CCEE",
            "#228833",
            "#CCBB44",
            "#EE6677",
            "#AA3377",
            "#BBBBBB",
        ]
        return result

    def get_all_differences(self, config_1, config_2):
        """
        Returns the differences in mean coverage between two configurations, for all modules,
        with information about significance.
        """

        config_1_results = self._storage[config_1]
        config_2_results = self._storage[config_2]
        common_keys = sorted(
            list(
                set(config_1_results.keys()).intersection(set(config_2_results.keys()))
            )
        )
        differences = []

        for module_name in tqdm.tqdm(common_keys):

            c1_module_results = np.array(config_1_results[module_name])
            c2_module_results = np.array(config_2_results[module_name])

            c1_results_at_1 = c1_module_results[:, -1]
            c2_results_at_1 = c2_module_results[:, -1]
            signficant_results = self.test_for_significant(
                {config_1: c1_results_at_1, config_2: c2_results_at_1}
            )
            difference_in_means_at_t = np.mean(
                [
                    c1_val - c2_val
                    for c1_val in c1_results_at_1
                    for c2_val in c2_results_at_1
                ]
            )  # np.mean(c1_results_at_1) - np.mean(c2_results_at_1)
            # assert difference_in_means_at_t ==
            stds_in_means_at_t = np.std(
                [
                    c1_val - c2_val
                    for c1_val in c1_results_at_1
                    for c2_val in c2_results_at_1
                ]
            )

            if len(signficant_results) > 0:
                if (config_1, config_2) in signficant_results:
                    # config 1 is significantly better than config 2
                    significance = f"{config_1} better"
                elif (config_2, config_1) in signficant_results:
                    # config 2 is significantly better than config 1
                    significance = f"{config_2} better"
                else:
                    significance = "not significant"
            else:
                significance = "not significant"

            differences.append(
                (
                    difference_in_means_at_t,
                    module_name,
                    significance,
                    stds_in_means_at_t,
                )
            )

        return sorted(differences)

    def get_mosa_increases_and_expansion_proportions(self):
        """
        Gets the increases of CODAMOSA over MOSA and the expansion proportions for each benchmark
        """

        def avg_percentage(num, denom):
            pcts = [
                num[i] / denom[i] if isinstance(denom[i], int) else 0
                for i in range(len(denom))
            ]
            return "{:.1f}%".format(np.mean(pcts) * 100)

        def avg_pct(num, denom):
            pcts = [
                num[i] / denom[i] if isinstance(denom[i], int) else 0
                for i in range(len(denom))
            ]
            return np.mean(pcts) * 100

        CM = "codamosa-0.8-uninterp"
        M = "mosa"
        diffs = self.get_all_differences(CM, M)
        ret_stuff = []
        for diff_in_means, module_name, sig, std_in_means in diffs:
            saved_llm_tests = self.get_values_of_stat(
                "LLMStageSavedTests", module_name, CM
            )
            expansion_llm_tests = self.get_values_of_stat(
                "LLMNeededExpansion", module_name, CM
            )
            uninterpreted_llm_tests = self.get_values_of_stat(
                "LLMNeededUninterpreted", module_name, CM
            )
            uninterpreted_calls_llm_tests = self.get_values_of_stat(
                "LLMNeededUninterpretedCallsOnly", module_name, CM
            )
            if sig != "not significant":

                print(
                    f"{module_name}: {diff_in_means}, expansion {avg_percentage(expansion_llm_tests,saved_llm_tests)}, uninterp {avg_percentage(uninterpreted_llm_tests, saved_llm_tests)}, uninterp calls {avg_percentage(uninterpreted_calls_llm_tests, saved_llm_tests)} "
                )
                ret_tuple = (
                    diff_in_means,
                    avg_pct(expansion_llm_tests, saved_llm_tests),
                    avg_pct(uninterpreted_llm_tests, saved_llm_tests),
                    avg_pct(uninterpreted_calls_llm_tests, saved_llm_tests),
                )
                ret_stuff.append(ret_tuple)
        return ret_stuff

    def get_query_time_and_saved_tests(self, module, tech):
        """
        Gets the average query time and number of saved tests for a module
        """
        saved_tests = np.array(
            [
                0 if val == "" else val
                for val in cc.get_values_of_stat("LLMStageSavedTests", module, tech)
            ]
        )
        query_time = np.array(
            [
                0 if val == "" else val
                for val in cc.get_values_of_stat("LLMQueryTime", module, tech)
            ]
        )
        return (
            np.mean(query_time),
            np.mean(saved_tests),
            np.mean(
                [
                    saved_tests[i] / query_time[i]
                    for i in range(len(saved_tests))
                    if query_time[i] > 0
                ]
            ),
        )

    def cycle_through_stat_correlations(self, config_1, config_2):
        # TODO
        diffs = cc.get_all_differences(config_1, config_2)
        for stat_name in stat_names:
            x_axis_stuff = []
            y_axis_stuff = []
            for diff in diffs:
                module = diff[1]
                coverage_diff = diff[0]
                config_1_stat_values = cc.get_values_of_stat(
                    stat_name, module, config_1
                )
                config_2_stat_values = cc.get_values_of_stat(
                    stat_name, module, config_2
                )
                print(config_1_stat_values)
                print(config_2_stat_values)
                stat_diffs = np.mean(
                    [
                        (c1 - c2)
                        for c1 in config_1_stat_values
                        for c2 in config_2_stat_values
                    ]
                )
                x_axis_stuff.append(coverage_diff)
                y_axis_stuff.append(stat_diffs)
            plt.scatter(x_axis_stuff, y_axis_stuff)
            plt.xlabel("avg diff in coverage")
            plt.ylabel("avg diff in " + stat_name)
            plt.show()

    def plot_significant_increases(self, configs=None):
        """
        Plots the results of the test_for_significant_final_cov() function.
        """
        if configs is None:
            configs = self.get_configs()
        configs_to_idx = {c: i for i, c in enumerate(sorted(configs))}
        winners_array = np.zeros(shape=(len(configs_to_idx), len(configs_to_idx)))
        # TODO: should we only look at common modules?
        modules = self.get_modules()
        for module in modules:
            # print(f'[{module}]')
            results = self.test_for_significant_final_cov(module, configs)
            for (config1, config2), (u, p) in results.items():
                idx1 = configs_to_idx[config1]
                idx2 = len(configs_to_idx) - configs_to_idx[config2] - 1
                winners_array[idx1, idx2] += 1
                # print(f"Adding 1 to ({idx1}, {idx2}) because {config1} had significantly higher coverage than {config2} (u={u}, p={p})")
        plt.imshow(winners_array, cmap="Blues", interpolation="nearest")
        plt.colorbar()
        for i in range(len(configs_to_idx)):
            for j in range(len(configs_to_idx)):
                if j != len(configs_to_idx) - i - 1:
                    plt.text(
                        j,
                        i,
                        s=int(winners_array[i, j]),
                        color="black" if winners_array[i, j] < 50 else "white",
                    )

        names_for_label = [get_nice_name(c) for c in configs_to_idx.keys()]

        plt.yticks(list(configs_to_idx.values()), labels=names_for_label)
        plt.xticks(
            list(configs_to_idx.values()),
            labels=list(reversed(names_for_label)),
            rotation=90,
        )
        plt.title(
            "Number of Benchmarks with Statistically Signifcant Coverage Increases"
        )
        plt.ylabel("Technique with Significantly Higher Coverage")
        plt.xlabel("Technique with Significantly Lower Coverage")
        plt.tight_layout()
        plt.show()

    def plot_all_codamosa_generations(self, benchmark, configs):
        """
        Plot all codamosa generation averages for benchmark, for each config in config
        """
        colors = [
            "#4477AA",
            "#66CCEE",
            "#228833",
            "#CCBB44",
            "#EE6677",
            "#AA3377",
            "#BBBBBB",
        ]
        colors = ["#DDAA33", "#BB5566", "#004488", "#000000"]
        for i, config in enumerate(configs):
            obsvs = self.get_codamosa_generations(benchmark, config)
            for obs in obsvs:
                plt.plot(obs, color=colors[i], alpha=0.3)
            plt.plot(np.mean(np.array(obsvs), axis=0), color=colors[i], label=config)
        plt.title(f"Number of Codex-Generated tests accepted for {benchmark}")
        plt.ylabel(f"Codex-generated tests saved")
        plt.xlabel(f"Time")
        plt.legend()
        plt.show()

    def plot_all_paths(self, benchmark, configs):
        """
        Plot all paths for benchmark, for each config in config
        """
        colors = [
            "#4477AA",
            "#66CCEE",
            "#228833",
            "#CCBB44",
            "#EE6677",
            "#AA3377",
            "#BBBBBB",
        ]
        colors = ["#DDAA33", "#BB5566", "#004488", "#000000"]
        for i, config in enumerate(configs):
            obsvs = self.get_observations(benchmark, config)
            for obs in obsvs:
                plt.plot(obs, color=colors[i], alpha=0.3)
            plt.plot(np.mean(np.array(obsvs), axis=0), color=colors[i], label=config)
        plt.title(f"Coverage Paths for {benchmark}")
        plt.ylabel(f"Proportion Branch + Line Coverage")
        plt.xlabel(f"Time")
        plt.legend()
        plt.show()

    def plot_average_over_all_benchmarks(self, configs=None):
        """
        Plots average coverage over all benchmarks.

        Kind of strange because most of the variance comes from the difference in base coverage between benchmarks
        """
        if configs is None:
            configs = self.get_configs()
        common_keys = self.get_common_modules(configs)
        all_average_coverage_per_config = {}
        # all_stdev_coverage_per_config = {}
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = list(prop_cycle.by_key()['color'])
        for i, config in enumerate(configs):
            # color = colors[i]
            coverage_results = []
            for module_name in common_keys:

                coverage_results.append(
                    np.mean(
                        np.array(self.get_observations(module_name, config)), axis=0
                    )
                )
            average_coverage = np.mean(np.array(coverage_results), axis=0)
            print(len(average_coverage))
            print(len(coverage_results))
            all_average_coverage_per_config[config] = average_coverage
        for config, coverage in all_average_coverage_per_config.items():
            plt.plot(coverage, label=config)  # , color=color)
            # plt.fill_between(range(len(coverage)), all_stdev_coverage_per_config[config] + coverage, coverage - all_stdev_coverage_per_config[config], alpha=0.2, color=color)
        plt.gca().yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0, decimals=0)
        )
        plt.ylabel("Proportion Branch + Line Coverage")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

    def plot_relative_average_per_benchmark(self, relative_config, configs):
        common_modules = [
            m
            for m in self.get_modules()
            if all(m in self._storage[c] for c in configs + [relative_config])
        ]

        all_differences: List[List[Tuple[float, float]]] = []
        for module_name in common_modules:
            base_results = [
                o[-1] for o in self.get_observations(module_name, relative_config)
            ]
            diffs_for_module: List[Tuple[float, float]] = []
            for i, config in enumerate(configs):
                coverage_results = [
                    o[-1] for o in self.get_observations(module_name, config)
                ]
                diffs = [c1 - c2 for c1 in coverage_results for c2 in base_results]
                avg_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                diffs_for_module.append((avg_diff, std_diff))
            all_differences.append(diffs_for_module)

        # all_differences[0] is the all differences for the first module
        # all_differences[0][0] is the difference + std for the first config, for the first modules
        # all_differences[0][0][0] is the difference for the first config, for the first modules
        try:
            base_idx = configs.index("codamosa-0.8-uninterp")
        except:
            base_idx = 0
        all_differences = sorted(all_differences, key=lambda d: d[base_idx][0])
        for i, config in enumerate(configs):
            diffs_and_stds = [
                diff_and_std_lst[i] for diff_and_std_lst in all_differences
            ]
            print(len(diffs_and_stds))
            plt.scatter(
                range(len(diffs_and_stds)),
                [diff for diff, std in diffs_and_stds],
                label=config,
            )

        # plt.fill_between(range(len(coverage)), all_stdev_coverage_per_config[config] + coverage, coverage - all_stdev_coverage_per_config[config], alpha=0.2, color=color)
        plt.gca().yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0, decimals=0)
        )
        plt.ylabel("Average difference in coverage at end vs MOSA")
        plt.xlabel("Modules")
        # plt.xticks([])
        plt.legend()
        plt.show()

    def multi_scatter_coverage(self, configs):
        """ """
        config_results = [self._storage[config] for config in configs]
        common_keys = sorted(
            [m for m in self.get_modules() if all(m in c for c in config_results)],
            key=lambda m: np.mean([l[-1] for l in config_results[0][m]]),
        )
        plt.figure(figsize=(20, 4))
        colors = [
            (68, 119, 170, 0.3),
            (104, 204, 238, 0.3),
            (34, 136, 51, 0.3),
            (204, 187, 68, 0.3),
            (238, 102, 119, 0.3),
            (170, 41, 119, 0.3),
            (187, 187, 187, 0.3),
        ]
        colors = [
            (68, 119, 170, 0.3),
            (204, 187, 68, 0.3),
            (238, 102, 119, 0.3),
            (170, 41, 119, 0.3),
            (104, 204, 238, 0.3),
            (34, 136, 51, 0.3),
            (187, 187, 187, 0.3),
        ]
        colors = [(c[0] / 256, c[1] / 256, c[2] / 256, c[3]) for c in colors]

        for i, config in enumerate(configs):
            data = []
            c = colors[i]
            for module_name in common_keys:
                data.append(np.mean([d[-1] for d in config_results[i][module_name]]))
            plt.scatter(range(len(data)), data, color=c, label=config)

        plt.ylabel("Average Coverage")
        plt.legend()
        plt.show()

    def violin_plot_coverage(self, configs):
        """ """
        config_results = [self._storage[config] for config in configs]
        all_common_keys = sorted(
            [m for m in self.get_modules() if all(m in c for c in config_results)],
            key=lambda m: np.mean(config_results[1][m]),
        )
        common_keys = []
        for key in all_common_keys:
            res = self.test_for_significant_final_cov(key)
            if len(res) > 0:
                common_keys.append(key)

        plt.figure(figsize=(20, 4))
        colors = [
            (68, 119, 170, 0.3),
            (104, 204, 238, 0.3),
            (34, 136, 51, 0.3),
            (204, 187, 68, 0.3),
            (238, 102, 119, 0.3),
            (170, 41, 119, 0.3),
            (187, 187, 187, 0.3),
        ]
        colors = [
            (68, 119, 170, 0.3),
            (204, 187, 68, 0.3),
            (238, 102, 119, 0.3),
            (170, 41, 119, 0.3),
            (104, 204, 238, 0.3),
            (34, 136, 51, 0.3),
            (187, 187, 187, 0.3),
        ]
        colors = [(c[0] / 256, c[1] / 256, c[2] / 256, c[3]) for c in colors]

        labels = []
        for i, config in enumerate(configs):
            data = []
            c = colors[i]
            for module_name in common_keys:
                data.append([d[0] for d in config_results[i][module_name]])
            violin_parts = plt.violinplot(data, positions=range(len(common_keys)))

            for partname in ("cbars", "cmins", "cmaxes"):
                vp = violin_parts[partname]
                vp.set_edgecolor((c[0], c[1], c[2], 1))
                vp.set_linewidth(1)

            for pc in violin_parts["bodies"]:
                pc.set_facecolor(c)
                pc.set_edgecolor((c[0], c[1], c[2], 1))

            labels.append(
                (mpatches.Patch(color=(c[0], c[1], c[2], 1)), get_nice_name(config))
            )
            # plt.boxplot(data, positions=range(len(common_keys)),notch=True, patch_artist=True,
            #     boxprops=dict(facecolor=c, color=c),
            #     capprops=dict(color=c),
            #     whiskerprops=dict(color=c),
            #     flierprops=dict(color=c, markeredgecolor=c),
            #     medianprops=dict(color=c),
        # 	showfliers=False)

        plt.legend(*zip(*labels))
        plt.ylabel("Average Coverage")
        plt.xticks([])
        plt.show()

    def plot_coverage_path_differences(self, config_1, config_2, plot_significant=True):
        """
        Plots the difference between mean coverage for config_1 - config_2
        """
        config_1_results = self._storage[config_1]
        config_2_results = self._storage[config_2]
        common_keys = sorted(
            list(
                set(config_1_results.keys()).intersection(set(config_2_results.keys()))
            )
        )

        for module_name in common_keys:
            if plot_significant:
                significant_diffs = self.test_for_significant_final_cov(module_name)
                alpha = 0.2
            else:
                significant_diffs = []
                alpha = 0.2
            # Get the mean results for each config
            config_1_mean = np.mean(np.array(config_1_results[module_name]), axis=0)
            config_2_mean = np.mean(np.array(config_2_results[module_name]), axis=0)
            difference = config_1_mean - config_2_mean
            if (config_1, config_2) in significant_diffs:
                plt.plot(difference, label=f"{module_name} (greater)")
            elif (config_2, config_1) in significant_diffs:
                plt.plot(difference, label=f"{module_name} (lesser)")
            else:
                plt.plot(difference, color="darkseagreen", alpha=alpha)
                # plt.plot(difference, label = f'{module_name}')

        plt.ylabel(f"Difference in Mean Coverage ({config_1} - {config_2})")
        plt.xlabel("Time")
        if plot_significant:
            plt.legend(title="Significant Differences at End", prop={"size": 8})
        plt.show()

    def plot_median_coverage_path_differences(self, config_1, config_2):
        """
        Plots the difference between median coverage for config_1 - config_2
        """
        config_1_results = self._storage[config_1]
        config_2_results = self._storage[config_2]
        common_keys = sorted(
            list(
                set(config_1_results.keys()).intersection(set(config_2_results.keys()))
            )
        )

        for module_name in common_keys:
            significant_diffs = self.test_for_significant_final_cov(module_name)
            # Get the median results for each config
            config_1_res = np.array(config_1_results[module_name])
            config_2_res = np.array(config_2_results[module_name])
            difference = []
            for i in range(config_1_res.shape[1]):
                config_1_vals = config_1_res[:, i]
                config_2_vals = config_2_res[:, i]
                all_diffs = []
                for val_1 in config_1_vals:
                    for val_2 in config_2_vals:
                        all_diffs.append(val_1 - val_2)
                difference.append(np.median(all_diffs))
            if (config_1, config_2) in significant_diffs:
                plt.plot(difference, label=f"{module_name} (greater)")
            elif (config_2, config_1) in significant_diffs:
                plt.plot(difference, label=f"{module_name} (lesser)")
            else:
                plt.plot(difference, color="darkseagreen", alpha=0.2)

        plt.ylabel(f"Difference in Median Coverage ({config_1} - {config_2})")
        plt.xlabel("Time")
        plt.legend(title="Significant Differences at End")
        plt.show()

    # Nice plots.

    def plot_all_differences_at_end(self, config_1, config_2, order_compared_to=None):
        """
        Plot all differendces at end, ordered.
        """
        c1_better_color = (0, 0.45, 0.7, 0.4)
        c2_better_color = (0.8, 0.4, 0, 0.4)
        no_better_color = (0.95, 0.90, 0.25, 0.4)
        color_map = {
            f"{config_1} better": c1_better_color,
            f"{config_2} better": c2_better_color,
            "not significant": no_better_color,
        }
        differences = self.get_all_differences(config_1, config_2)
        if order_compared_to is not None:
            module_order_comparatively = [
                d[1] for d in self.get_all_differences(config_1, order_compared_to)
            ]
            avail_modules = [d[1] for d in differences]
            module_to_diff = {d[1]: d for d in differences}
            differences = [
                module_to_diff[module]
                for module in module_order_comparatively
                if module in avail_modules
            ]
        plt.axhline(0, color="grey", linestyle="dashed", linewidth=1, zorder=0)
        plt.scatter(
            range(len(differences)),
            [d[0] for d in differences],
            color=[color_map[d[2]] for d in differences],
            s=4,
            zorder=10,
        )
        plt.xticks()
        plt.ylabel("Absolute difference in mean coverage at end")
        plt.xlabel("Modules ordered by differences in average coverage")
        plt.gca().yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0, decimals=0)
        )
        for color in color_map.values():
            plt.fill_between(
                range(len(differences)),
                [d[0] - d[3] for d in differences],
                [d[0] + d[3] for d in differences],
                where=[color_map[d[2]] == color for d in differences],
                color=color,
                alpha=0.2,
            )

        legend_lines = [
            Line2D(
                [0],
                [0],
                markerfacecolor=(c1_better_color),
                marker="o",
                color="w",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                markerfacecolor=(c2_better_color),
                marker="o",
                color="w",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                markerfacecolor=(no_better_color),
                marker="o",
                color="w",
                markersize=8,
            ),
        ]

        plt.legend(
            legend_lines,
            [
                f"Significant: {config_1} > {config_2}",
                f"Significant: {config_1} < {config_2}",
                "No significant difference",
            ],
            loc="best",
        )

        plt.show()

    def scatter_plot_coverage(
        self, config_1, config_2, plot_significant=True, coverage_type="TOTAL"
    ):
        """
        A comparative scatter plot of the coverage of config_1 and config_2.
        """
        config_1_results = self._storage[config_1]
        config_2_results = self._storage[config_2]
        common_keys = self.get_common_modules(
            [config_1, config_2]
        )  # sorted(list(set(config_1_results.keys()).intersection(set(config_2_results.keys()))))
        print("Total num benchmarks:", len(common_keys))
        # keys that are in config_1_results but not in config_2_results
        # c1_has_c2_does_not = sorted(list(set(config_1_results.keys()).difference(set(config_2_results.keys()))))
        # c2_has_c1_does_not = sorted(list(set(config_2_results.keys()).difference(set(config_1_results.keys()))))
        # print(f'Num benchmarks in {config_1} but not in {config_2}:', len(c1_has_c2_does_not))
        # print(c1_has_c2_does_not)
        # print(f'Num benchmarks in {config_2} but not in {config_1}:', len(c2_has_c1_does_not))
        # print(c2_has_c1_does_not)
        c1_nosig = []
        c2_nosig = []
        c1_c1gtc2 = []
        c2_c1gtc2 = []
        c1_c2gtc1 = []
        c2_c2gtc1 = []
        plt.figure(figsize=(3.2, 2.4))
        for module_name in common_keys:
            if coverage_type == "TOTAL":
                data_1 = [l[-1] for l in self.get_observations(module_name, config_1)]
                data_2 = [l[-1] for l in self.get_observations(module_name, config_2)]
            elif coverage_type == "BRANCH":
                data_1 = self.get_values_of_stat(
                    "BranchCoverage",
                    module_name,
                    config_1,
                )
                data_2 = self.get_values_of_stat(
                    "BranchCoverage",
                    module_name,
                    config_2,
                )
            elif coverage_type == "LINE":
                data_1 = self.get_values_of_stat(
                    "LineCoverage",
                    module_name,
                    config_1,
                )
                data_2 = self.get_values_of_stat(
                    "LineCoverage",
                    module_name,
                    config_2,
                )
            else:
                print("Invalid coverage type: ", coverage_type)
                return
            config_1_mean_end = np.mean(data_1)
            config_2_mean_end = np.mean(data_2)
            if plot_significant:
                significant_diffs = self.test_for_significant(
                    {config_1: data_1, config_2: data_2}
                )
                if (config_1, config_2) in significant_diffs:
                    c1_c1gtc2.append(config_1_mean_end)
                    c2_c1gtc2.append(config_2_mean_end)
                elif (config_2, config_1) in significant_diffs:
                    c1_c2gtc1.append(config_1_mean_end)
                    c2_c2gtc1.append(config_2_mean_end)
                else:
                    c1_nosig.append(config_1_mean_end)
                    c2_nosig.append(config_2_mean_end)
            else:
                c1_nosig.append(config_1_mean_end)
                c2_nosig.append(config_2_mean_end)
        print(f"{config_1} > {config_2} on {len(c1_c1gtc2)} benchmarks")
        print(f"{config_2} > {config_1} on {len(c1_c2gtc1)} benchmarks")
        plt.scatter(
            c1_nosig,
            c2_nosig,
            marker="o",
            linewidths=0.5,
            facecolors=(1, 1, 1, 0.6),
            edgecolors=not_sig_color(0.6),
            s=12,
            label="No significant difference",
        )
        if len(c1_c1gtc2) > 0:
            plt.scatter(
                c1_c1gtc2,
                c2_c1gtc2,
                linewidth=1,
                marker=get_marker(config_1),
                color=get_color(config_1, 0.8),
                s=36,
                label=f'Significant: {get_nice_name(config_2)} < {get_nice_name(config_1).replace("CodaMOSA-","")}',
            )
        if len(c1_c2gtc1) > 0:
            plt.scatter(
                c1_c2gtc1,
                c2_c2gtc1,
                linewidth=1,
                marker=get_marker(config_2),
                color=get_color(config_2, 0.9),
                s=36,
                label=f'Significant: {get_nice_name(config_2)} > {get_nice_name(config_1).replace("CodaMOSA-","")}',
            )

        plt.gca().xaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0, decimals=0)
        )
        plt.gca().yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0, decimals=0)
        )
        plt.xlabel(f"{get_nice_name(config_1)} Average Coverage", fontsize=8)
        plt.ylabel(f"{get_nice_name(config_2)} Average Coverage", fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6, rotation=90)

        plt.legend(prop={"size": 6})
        plt.tight_layout(pad=0.6, h_pad=0.6, w_pad=0.2)

    def plot_coverage_path_differences_highlight_significant(self, config_1, config_2):
        """
        Plots the difference between the mean coverage of two configurations, highlighting
        the period of time where the coverage was significantly different.
        """

        config_1_results = self._storage[config_1]
        config_2_results = self._storage[config_2]
        common_keys = sorted(list(self.get_common_modules([config_1, config_2])))
        # common_keys = random.choices(common_keys, k=20)

        plt.figure(figsize=(5, 2.8))

        max_y_value = 0
        min_y_value = 0
        segments = []
        colors = []
        linestyles = []
        c1_better_color = get_color(config_1, 0.4)  # (0, 0.45, 0.7, 0.4)
        c2_better_color = get_color(config_2, 0.4)  # (0.8, 0.4, 0, 0.4)
        no_better_color = not_sig_color(0.4)
        c1_better_color = (79 / 256, 194 / 256, 219 / 256, 0.8)
        no_better_color = (49 / 256, 99 / 256, 137 / 256, 0.8)
        c2_better_color = (19 / 256, 4 / 256, 54 / 256, 0.8)

        def append_current_segment(segment, segment_type, extend_value=None):
            """
            Helper function to append the current segment to the appropriate list.
            If extend_value is not None, interpolate the line between the last value
            of segment and the extend_value
            """
            if extend_value is not None:
                connect_time = segment[-1][0] + 0.5
                segment.append((connect_time, (extend_value + segment[-1][1]) / 2))
                new_segment = [(connect_time, (extend_value + segment[-1][1]) / 2)]
            else:
                new_segment = []

            if segment_type == "c1_better":
                segments.append(segment)
                colors.append(c1_better_color)
                linestyles.append("solid")
            elif segment_type == "c2_better":
                segments.append(segment)
                colors.append(c2_better_color)
                linestyles.append((0, (5, 1)))
            elif segment_type == "neither":
                segments.append(segment)
                colors.append(no_better_color)
                linestyles.append((0, (1, 3)))

            return new_segment

        for module_name in tqdm.tqdm(common_keys):

            c1_module_results = np.array(config_1_results[module_name])
            c2_module_results = np.array(config_2_results[module_name])
            end_time = c1_module_results.shape[1]

            cur_segment: List[Tuple[int, float]] = []
            last_segment_type = None
            cur_segment_type = None

            for t in range(end_time):
                c1_results_at_1 = c1_module_results[:, t]
                c2_results_at_1 = c2_module_results[:, t]
                signficant_results = self.test_for_significant(
                    {config_1: c1_results_at_1, config_2: c2_results_at_1}
                )
                difference_in_means_at_t = np.mean(c1_results_at_1) - np.mean(
                    c2_results_at_1
                )
                if difference_in_means_at_t < min_y_value:
                    min_y_value = difference_in_means_at_t
                if difference_in_means_at_t > max_y_value:
                    max_y_value = difference_in_means_at_t

                if len(signficant_results) > 0:
                    if (config_1, config_2) in signficant_results:
                        # config 1 is significantly better than config 2
                        cur_segment_type = "c1_better"
                    elif (config_2, config_1) in signficant_results:
                        # config 2 is significantly better than config 1
                        cur_segment_type = "c2_better"
                    else:
                        # this should not happen
                        print(module_name, signficant_results)
                        assert False
                else:
                    # no significant difference
                    cur_segment_type = "neither"

                # In the first iteration, set the last segment type to the current segment type
                if last_segment_type is None:
                    last_segment_type = cur_segment_type
                    cur_segment.append((t, difference_in_means_at_t))
                    continue

                # Append the current segment if it is different from the last segment, reset the current segment
                if (
                    cur_segment_type != last_segment_type
                    and last_segment_type is not None
                ):
                    if len(cur_segment) > 0:
                        cur_segment = append_current_segment(
                            cur_segment, last_segment_type, difference_in_means_at_t
                        )
                    else:
                        print(f"Empty segment for module {module_name}")
                        assert False
                # Append to the current segment
                cur_segment.append((t, difference_in_means_at_t))
                # Set the segment type to the current segment type
                last_segment_type = cur_segment_type

            # Append the last segment
            if len(cur_segment) > 0:
                append_current_segment(cur_segment, last_segment_type, None)

        lc = mcollections.LineCollection(
            segments, colors=colors, linewidth=1, linestyles=linestyles
        )

        def opaque(color):
            return (color[0], color[1], color[2], 1)

        legend_lines = [
            Line2D(
                [0], [0], color=opaque(c1_better_color), linewidth=1, linestyle="solid"
            ),
            Line2D(
                [0],
                [0],
                color=opaque(c2_better_color),
                linewidth=1,
                linestyle=(0, (5, 1)),
            ),
            Line2D(
                [0],
                [0],
                color=opaque(no_better_color),
                linewidth=1,
                linestyle=(0, (1, 3)),
            ),
        ]

        plt.gca().add_collection(lc)
        plt.xlim(-1, end_time + 1)
        plt.xlabel("Time (s)")
        plt.ylim(min_y_value - 0.1, max_y_value + 0.1)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel(
            f"Per-Benchmark Avg. Coverage Diff.\n{get_nice_name(config_1)} - {get_nice_name(config_2)}",
            fontsize=9.5,
        )
        plt.gca().yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0, decimals=0)
        )
        plt.legend(
            legend_lines,
            [
                f"Significant: {get_nice_name(config_1)} > {get_nice_name(config_2)}",
                f"Significant: {get_nice_name(config_1)} < {get_nice_name(config_2)}",
                "No significant diff.",
            ],
            prop={"size": 7.5},
            ncol=2,
            bbox_to_anchor=(0.98, 1.1),
        )
        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.2)


def add_codamosa_generations_from_file(
    statistics_file: str, cc: CoverageContainer, module_name: str, config_name: str
):
    """
    Adds the codamosa_generations data from `statistics_file` to the CoverageContainer `cc`.
    """
    if os.path.isfile(statistics_file):
        with open(statistics_file, "r") as f:
            time_absorbed_tuples = [l.split(",") for l in f.readlines()]
            times = [int(tup[0]) for tup in time_absorbed_tuples]
            num_test_cases = [int(tup[1]) for tup in time_absorbed_tuples]
            times_idx = 0
            last_value = 0
            interp_num_test_cases = []
            for time in range(1, MAX_TIME + 1):
                cur_time = (
                    times[times_idx] if times_idx < len(times) else MAX_TIME + 1000
                )
                if time < cur_time:
                    interp_num_test_cases.append(last_value)
                elif time == cur_time:
                    last_value = num_test_cases[times_idx]
                    interp_num_test_cases.append(last_value)
                    times_idx += 1

            cc.add_codamosa_generations_for_module(
                module_name, config_name, interp_num_test_cases
            )


def add_coverage_from_file(
    statistics_file: str, cc: CoverageContainer, module_name: str, config_name: str
):
    """
    Adds the coverage data from `statistics_file` to the CoverageContainer `cc`.
    """
    if os.path.isfile(statistics_file):
        with open(statistics_file, "r") as f:
            headers = f.readline().strip().replace('"', "").split(",")
            data = f.readline().strip().replace('"', "").split(",")
            if len(headers) != len(data):
                print(f"error in {statistics_file}, only {len(data)} data points")
                return
            coverage_start_idx = 0
            while not headers[coverage_start_idx].startswith("CoverageTimeline_T"):
                coverage_start_idx += 1
            coverage_data = np.array([float(x) for x in data[coverage_start_idx:]])
            cc.add_observation_for_module(module_name, config_name, coverage_data)


def add_other_stats_from_file(
    statistics_file: str, cc: CoverageContainer, module_name: str, config_name: str
):
    """
    Adds the other from `statistics_file` to the CoverageContainer `cc`.
    """
    if os.path.isfile(statistics_file):
        with open(statistics_file, "r") as f:
            headers = f.readline().strip().replace('"', "").split(",")
            data = f.readline().strip().replace('"', "").split(",")
            if len(headers) != len(data):
                print(f"error in {statistics_file}, only {len(data)} data points")
                return
            other_stats = {
                headers[i]: data[i]
                for i in range(len(headers))
                if not headers[i].startswith("CoverageTimeline_T")
            }
            for stat_name, data in other_stats.items():
                cc.add_other_stat_for_module(module_name, config_name, stat_name, data)


def main(input_dir: str):
    """
    Collects all the data in `input_dir` into a CoverageContainer so it can be accessed and analyzed.
    """
    cc = CoverageContainer()
    for config_name in os.listdir(input_dir):
        if config_name == "run-data":
            continue
        config_dir = os.path.join(input_dir, config_name)
        if os.path.isdir(config_dir):
            print(f"retrieving info from {config_dir}")
            for folder_name in tqdm.tqdm(os.listdir(config_dir)):
                module_dir = os.path.join(config_dir, folder_name)
                module_name = folder_name.split("-")[0]
                if os.path.isdir(module_dir):
                    statistics_file = os.path.join(module_dir, "statistics.csv")
                    add_coverage_from_file(
                        statistics_file, cc, module_name, config_name
                    )
                    add_other_stats_from_file(
                        statistics_file, cc, module_name, config_name
                    )
                    codamosa_timeline_file = os.path.join(
                        module_dir, "codamosa_timeline.csv"
                    )
                    add_codamosa_generations_from_file(
                        codamosa_timeline_file, cc, module_name, config_name
                    )
    return cc


def plot_main(cc: CoverageContainer):
    print(f"Analyzing data, making comparisons with minimum number of reps {MIN_REPS}")
    print("=======Making Figure 2(a)========")
    cc.plot_coverage_path_differences_highlight_significant(
        "codamosa-0.8-uninterp", "mosa"
    )
    if PLOTS_DIR is not None:
        plt.savefig(os.path.join(PLOTS_DIR, f"vs_mosa_paths.pdf"))
    else:
        plt.show()
    print("=======Making Figure 2(b)========")
    cc.plot_coverage_path_differences_highlight_significant(
        "codamosa-0.8-uninterp", "codex-only"
    )
    if PLOTS_DIR is not None:
        plt.savefig(os.path.join(PLOTS_DIR, f"vs_codex_paths.pdf"))

    else:
        plt.show()
    other_configs = [
        ("mosa", "a"),
        ("codex-only", "b"),
        ("codamosa-0.8", "c"),
        ("codamosa-0.2-uninterp", "d"),
        ("codamosa-0.8-uninterp-no-targeting", "e"),
        ("codamosa-0.8-uninterp-small", "f"),
    ]
    for config_name, subfig_id in other_configs:
        print(f"=======Making Figure Y({subfig_id})========")
        cc.scatter_plot_coverage(config_name, "codamosa-0.8-uninterp")
        if PLOTS_DIR is not None:
            plt.savefig(
                os.path.join(
                    PLOTS_DIR,
                    f'vs_{get_nice_name(config_name).replace("CodaMOSA-", "")}_scatter.pdf',
                )
            )
        else:
            plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} data-dir-or-pickle [fig_output_dir]")
        exit(1)

    if len(sys.argv) == 3:
        PLOTS_DIR = sys.argv[2]
        if not os.path.isdir(PLOTS_DIR):
            print(f"{PLOTS_DIR} is not a valid directory")
            exit(1)

    if os.path.isdir(sys.argv[1]):
        cc = main(sys.argv[1])
    elif os.path.isfile(sys.argv[1]):
        cc = pickle.load(open(sys.argv[1], "rb"))
    else:
        print(
            "Unsupported input method; either give directories as argument or a single pickle file"
        )
        exit(1)

    if not sys.flags.interactive:
        plot_main(cc)


## Analysis for flutils.packages test study
# cm = 'codamosa-0.8-uninterp'
# data = {mod: [o[-1] for o in cc.get_observations(mod, cm)]for mod in cc.get_modules()}
# cc.test_for_significant(data)
# avgs = {mod: np.mean(mod_data) for mod,mod_data in data.items()}
