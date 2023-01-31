"""
Creates the cumulative similarities plot from pre-processed similarity data. 
"""
import os
import pickle
import sys
from itertools import cycle
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# lines = ["-","--","-.",":", (0, (3,1,1,1,1,1))]
lines = ["-", "--", ":"]
markers = ["x", "d", "+", "o"]
colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#333333"]
colors = [
    "#544F7D",
    "#695F77",
    "#7F6E72",
    "#947E6C",
    "#AA8D67",
    "#BF9D61",
    "#D4AC5B",
    "#EABC56",
    "#FFCB50",
]
colors = ["#544F7D", "#8D786E", "#C6A25F", "#FFCB50", "#BC9850", "#786450", "#353150"]
colors = [e for a in colors for e in [a] * 3]
colors = ["#012849", "#5E7C45", "#BAD040"]
colors = [e for a in colors for e in [a] * 9]
# colors = ["#021E36", "#445F3B", "#87A040", "#C9E145"]
linecycler = cycle(lines)
colorcycler = cycle(colors)
markercycler = cycle(markers)


def hl_zorder(proj_name):
    if proj_name in {"apimd", "sty", "flutes", "thonny", "flutils"}:
        return 40
    else:
        return 0


def hl_color(proj_name, col):
    if proj_name in {"apimd", "sty", "flutes", "thonny", "flutils"}:
        return "black"
    else:
        return col


def hl_op(proj_name):
    if proj_name in {"apimd", "sty", "flutes", "thonny", "flutils"}:
        return 1
    else:
        return 0.6


def hl_width(proj_name):
    if proj_name in {"apimd", "sty", "flutes", "thonny", "flutils"}:
        return 1.5
    else:
        return 1


def plot_cumulative_similarities(max_sim_list: List[float], proj_name=""):
    # A list of similarity, number of generations under that similarity
    cumulative_sims = []
    sims_under_current = 0
    last_sim = 0
    for max_sim in max_sim_list:
        if max_sim > last_sim:
            cumulative_sims.append((last_sim, sims_under_current))
            last_sim = max_sim
        sims_under_current += 1
    if last_sim > cumulative_sims[-1][0]:
        cumulative_sims.append((last_sim, sims_under_current))
    if last_sim == 1:
        print(
            f"{proj_name}: {cumulative_sims[-1][1]-cumulative_sims[-2][1]} ({(100*(cumulative_sims[-1][1]-cumulative_sims[-2][1])/cumulative_sims[-1][1])} %) have similarity {last_sim}"
        )
    # backfill for those where sparse data points at the end
    if cumulative_sims[-1][0] - cumulative_sims[-2][0] > 0.1:
        second_last = cumulative_sims[-2][0]
        last = cumulative_sims[-1][0]
        cumulative_sims = (
            cumulative_sims[:-2]
            + [(r, cumulative_sims[-2][1]) for r in np.arange(second_last, last, 0.05)]
            + cumulative_sims[-1:]
        )

    if proj_name == "":
        plt.plot(
            [cs[0] for cs in cumulative_sims],
            [cs[1] / sims_under_current for cs in cumulative_sims],
        )
    else:
        plt.plot(
            [cs[0] for cs in cumulative_sims],
            [cs[1] / sims_under_current for cs in cumulative_sims],
            color=hl_color(proj_name, next(colorcycler)),
            label=proj_name.replace("python-", ""),
            linestyle=next(linecycler),
            mfc="white",
            marker=next(markercycler),
            markevery=0.05,
            markersize=6,
            linewidth=hl_width(proj_name),
            drawstyle="steps-post",
            alpha=hl_op(proj_name),
            zorder=hl_zorder(proj_name),
        )
    bot, top = plt.ylim()
    plt.ylim((0, top))
    ax = plt.gca()

    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1))


def plot_edit_dist_similarities(similarities_map, proj_name_order, plot_output_dir):
    plt.figure(figsize=(6.4, 4))
    plt.yticks(fontsize=9, rotation=90)
    for proj_name in proj_name_order:
        sorted_similarities = similarities_map[proj_name]
        plot_cumulative_similarities(sorted_similarities, proj_name)
    plt.xlabel(f"Maximum Similarity ($S_e=1-$normalized edit distance)")
    plt.ylabel(
        "Cumulative Percent of Test Cases\nwith Maximum Similarity $\leq S_e$",
        fontsize=10,
    )
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, "editdistance_similarity.pdf"))
    plt.show()


def summarize_duplicates(data):
    print("=====Summary of exactly duplicated tests=====")
    total_duplicates = 0
    passes = 0
    import collections

    for proj in data:
        similarities, lens, test_pairs = data[proj]
        project_counter = collections.Counter()
        for i, sim in enumerate(similarities):
            if sim == 1:
                project_counter[test_pairs[i][0]] += 1
        if len(project_counter) > 0:
            print(f"===={proj}====")
            for elem, count in project_counter.items():
                print(f">> {count} occurrences of <<\n{elem}")
                total_duplicates += count
                if elem == "pass\n":
                    passes += count
    print(
        f"=================\n{passes}/{total_duplicates} ({round(100*passes/total_duplicates,1)}%) of duplicated tests are 'pass'"
    )


def scatter_sizes(data):
    colorcycler = cycle(colors)
    for proj in reversed(data):
        similarities, lens, _ = data[proj]
        plt.scatter(
            similarities,
            lens,
            color=next(colorcycler),
            marker=next(markercycler),
            alpha=0.2,
            label=proj.replace("python-", ""),
        )
    plt.xlabel("Maximum Similarity ($S_e$)")
    plt.ylabel("Test Case Length (characters)")
    plt.legend(ncol=2, fontsize=7, loc="upper right")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if len(sys.argv) != 3:
        print(
            f"usage: python3 {sys.argv[0]} pkl_file_with_similarity_analysis_data plots_output_dir"
        )
        exit(1)

    # Second processing part
    # Contains the similarity maps, the len, and the actual most similar test cases.
    all_data_file = open(sys.argv[1], "rb")
    data = pickle.load(all_data_file)
    similarities_map = {proj: sorted(v[0]) for proj, v in data.items()}
    all_data_file.close()

    sim_tuples = [(k, v) for k, v in similarities_map.items()]
    sim_tuples = sorted(
        sim_tuples,
        reverse=True,
        key=lambda proj_sims: (
            proj_sims[1][-1],
            len([v for v in proj_sims[1] if v == proj_sims[1][-1]]) / len(proj_sims[1]),
        ),
    )
    extra_tuples = [
        (p, (sims[-1], len([v for v in sims if v == sims[-1]]) / len(sims)))
        for p, sims in sim_tuples
    ]

    print(extra_tuples[-10:])
    proj_name_order = [st[0] for st in sim_tuples]
    plot_edit_dist_similarities(similarities_map, proj_name_order, sys.argv[2])
    summarize_duplicates(data)
    # scatter_sizes(data)
