# CodaMOSA Artifact

This repo contains code for replication/data analysis of the data from the evaluation of CodaMOSA for ICSE'23. The raw dataset can be found in the [dataset repository](https://github.com/microsoft/codamosa-dataset).

## License

The code in this subfolder is all MIT licensed.

## Directory structure

```
- README.md: this file 
- scripts:
    - start_benchmark_container.sh: starts and attaches to the benchmark docker
         container, creating a docker container `codamosa-benchmarks-container`,
         which clones all the test repositories into the test-apps subdirectory
    - create_and_export_figures.py: creates all the figures from the `final-exp` 
         directory, and exports these to a directory of your specification.
    - plot_similarities.py: creates Fig. 4, exports into a specified directory.
    - run_one.sh: lets you run a new instance of CodaMOSA on a given module
    - create_similarity_data.py: runs the similarity analysis (to re-create
          similarity_analysis_info.pkl)
    - modules_base_and_name.csv: ordered csv of module base directory and name,
          for use by create_similarity_data.py
- config_args:
    - each file contains the arguments used to run a particular configuration 
         of CodaMOSA, as demonstrated in the paper. 
- processed-data: data processed for plotting
    - big-cc-final.pkl: main evaluation data, processed for plotting
    - similarity_analysis_info.pkl: similarity analysis data, processed for plotting
- docker-images:
    - benchmarks-docker.tar.gz: a saved docker image, containing scripts to set up the 
      benchmarks, and preliminary experiment results used to filter those benchmarks
    - codamosa-docker.tar.gz: a saved docker image, containing the Pynguin runner that
      allows you to run CodaMOSA/MOSA.
```

## Setup

Software requirements: `docker`, `bash`

Begin by loading the two images:
```
$ docker load < benchmarks-docker.tar.gz  
$ docker load < codamosa-docker.tar.gz
```
Depending on how docker is installed on your system, you may need to use `sudo docker <CMD>`.

This should leave you with two new docker images:
```
$ docker images 
REPOSITORY             TAG       IMAGE ID       CREATED          SIZE
benchmarks-image       latest    ...
pynguin-runner         latest    ...
...
```

**Note**: Many of the following scripts will invoke `docker` commands. You may encounter issues running them if your user is not in the [docker group](https://docs.docker.com/engine/install/linux-postinstall/). To resolve this, you may either follow the instructions from docker to add your user to the docker group (preferred), or preface the commands below with `sudo`, or add `--user=root` as an option to the docker commands in the scripts below. 

### Start the benchmark container

Start the benchmark container with
```
$ ./scripts/start_benchmark_container.sh
```
This will clone all the directories into the `test-apps` subdirectory (around 22 minutes) of the root replication folder, and create a `codamosa-benchmarks-container` where you can examine the benchmarks and results of the preliminary testing. At the end of the run it prints how we filtered the original number of main project benchmarks down to the 486 modules used in our experiments:
```
Total number of benchmarks in main project:
2782
Benchmarks on which we are able to run:
1541
Over the following number of  modules:
27
Benchmarks which have 1 coverage at end:
115
Benchmarks which have 1 coverage at start:
106
Benchmarks which have <1 coverage at end for at least one run
1430
Take at most 20 from each set sharing the same parent module
526
Number of those modules which have to be timed out aggressively
39
A module we will find later doesn't consistently produce a statistics.csv file, one on which MOSA 'fails to run' (one of the two runs in the smoke test fail)
1
Final number of modules:
486
```


Not all of the projects cloned in the benchmark container actually end up having any modules in our benchmark set. These include some of the long-to-clone and large projects (`pandas`, `matplotlib`, etc.). If you just want to set up the projects from which modules are successfully found,  start the benchmark container with:
```
$ ./scripts/start_small_benchmark_container.sh
```
This takes around 14 minutes to run; a few errors will be thrown when running `pipreqs` on the uncloned modules (`pandas`, `matplotlib`, etc.).

Either way, you can now view the lists of potential test modules and the list of modules used as benchmarks in the paper (`test-apps/good_modules.csv`):
```
$ wc -l test-apps/all_potential_modules.csv
     2789 test-apps/all_potential_modules.csv  # or 1784 if you only started the small benchmark container
$ wc -l test-apps/good_modules.csv
     486 test-apps/good_modules.csv
```
Both CSVs have two columns: the first gives the root directory for the Python project, the second the fully qualified module name from that root directory. 

## Replicating figures from the paper, and analyzing data.

**Requirements:** A python3 installation with the following packages: `matplotlib`, `numpy`, `scipy`, `tqdm`

### Main Evaluation Plots

Run the following: 
```
$ python3 scripts/create_and_export_figures.py processed-data/big-cc-final.pkl PLOTS_OUTPUT_DIR
```
where `PLOTS_OUTPUT_DIR` is where you would like the plots to go. This will create the paths through time plots and scatter plots in the paper. Because the coverage differences throug times plots involve the computation of significance at each time, they take a bit longer to generate, around 1m45s each.

To validate that the data in `big-cc-final.pkl` is consistent with the raw `final-exp` results, you may clone [the raw dataset](https://github.com/microsoft/codamosa-dataset). Unzip the benchmarks within that dataset, and run the following, where `FINAL_EXP_LOC` points to the cloned `codamosa-dataset` directory.
```
$ python3 scripts/create_and_export_figures.py FINAL_EXP_LOC/final-exp PLOTS_OUTPUT_DIR
```
This will create a fresh new coverage container (`cc`) containing the data in `final-exp`, but be aware it takes a while to process all the information (approximately 12 minutes). 

To conduct exploratory data analysis, run in interative mode
```
$ python3 -i scripts/create_and_export_figures.py processed-data/big-cc-final.pkl PLOTS_OUTPUT_DIR
```
The `cc` object contains processed data, as well as several methods for exploratory data analysis and plotting. If running from `final-exp`, you can pickle the created `cc` object and verify that it has identical contents to the `big-cc-final.pkl` file.

### Similarity Analysis and Plots

**Requirements:** A python3 installation with the following packages: `astor`, `editdistance`, `matplotlib`, `tqdm`

To generate the plot in Figure 4, from pre-processed data, run
```
$ python3 scripts/plot_similarities.py processed-data/similarity_analysis_info.pkl PLOTS_OUTPUT_DIR
```

Running the full similarity analysis takes a *very long time*. The particularly long part is to run the analysis for all the `ansible` files (14-17 hours). The provided `scripts/modules_base_and_name.csv` places these modules last. You can remove some of the modules from this file to restrict the length of the similarity analysis. You can then re-run the analysis from raw data to generate the processed data as follows:
```
$ python3 scripts/create_similarity_data.py test-apps FINAL_EXP_LOC/final-exp/codamosa-0.8-uninterp scripts/modules_base_and_name.csv OUTPUT_PKL_FILE
```
Where  `OUTPUT_PKL_FILE` is where you want to store the resulting pickle file.  

## Running CodaMOSA

First create the benchmarks container as above.

Make sure you have loaded the `codamosa-runner` image. You can use `scripts/run_one.sh` invoke the runner as follows.

```
$ ./scripts/run_one.sh MODULE_NAME OUT_DIR ARGS_FILE AUTH_KEY SEARCH_TIME
```

`./scripts/run_one.sh` takes the module name, finds its base directory by grepping against the `test-apps/good_modules.csv` file, then creates the output directory, and invokes `docker run codamosa-runner` with the provided arguments and authorization key (or existing codex generations for replaying). It also creates the appropriate bind mounts for the runner container.

For example, try
```
$ ./scripts/run_one.sh flutils.packages /tmp/flutils config-args/codamosa-0.8-uninterp 30 $PLAY_OPTION
```
On most runs, this should invoke a the targeted generation step once, resulting in 10 calls to Codex. 

Note (Jun 2024): the files in `config-args` use the model `code-davinci-002`. This model is no longer accessible, to make new queries to OpenAI you will have to update it to a different model name. For historical reasons (since CodaMOSA was evaluated with `code-davinci-002`, we will not update the files in `config-args` to a new default.


For the `$PLAY_OPTION`, you have two options:


#### Re-running with an OpenAI authorization key
The file passed as `AUTH_KEY_FILE` should contain your OpenAI key. The `$PLAY_OPTION` should look like this:
```
--auth [PATH_TO_YOUR_AUTH_KEY]
```
The configuration arguments by default use the model `code-davinci-002`; if your authorization key does not give access to that model you may need to change it. If you change to a model that has a smaller context size (not 4000), you will need to modify the CodaMOSA code to reflect this (`pynguin.languagemodels.model`), and rebuild the runner following the instructions in the codamosa directory.


#### Re-running with a generation file
If you do not have access to an OpenAI authorization key, you can use the `--replay` argument with a suitable input. The expected format is that of a `codex_generations.py` file, as can be found in one of the CodaMOSA or CodexOnly runs in the `final-exps` directory of the [CodaMOSA dataset](https://github.com/microsoft/codamosa-dataset). 

Copy the generation file in your output directory, i.e. `OUTPUT_DIR/previous_gens.py`

The `$PLAY_OPTION` should look like this:
```
--replay [PATH_TO_PREVIOUS_CODEX_GENERATIONS]
```
(the `run_one.sh` script names the output directory `/output`, thus the use of this as the root)

*Note:* Codamosa can write to the output directory, so if you are testing a module that can affect the file system, make sure to backup the contents of `OUTPUT_DIR/previous_gens.py` somewhere else!

### Running CodaMOSA on your own module.

Base a `docker run` command off of the run command in `run_one.sh`. 

The arguments following `docker run codamosa-runner` are the regular arguments to Pynguin/CodaMOSA. For the docker arguments, ensure that your test app directory `your_app_dir` contains a package.txt file, in the format of a requirements.txt, which lists any python dependencies to install via pip. Create mounts to:
- `-v your_output_dir:/output`
- `-v your_app_dir:/input:ro`
- `-v your_app_dir:/package:ro`
and pass the module name to `--module-name`.


### Clean up
If you have run `./scripts/start_benchmark_container.sh`, you can remove the created container as follows:

```
$ docker container stop codamosa-benchmarks-container     # unless it is already stopped
$ docker container rm codamosa-benchmarks-container
```



## FAQ

*Q. Why are there two docker containers?*

*A.* MOSA and CodaMOSA, as random testing algorithms, may cause damage to the file system we are running them in. Thus, we sandbox the runs in a docker container, as was done to evaluate MOSA in Pynguin. The other docker container contains the main experimental benchmarks.

*Q. How did you populate all_potential_modules.csv and package.txt for each module?*

*A.*  This is described in the README of the benchmarks docker container, but in short, there are 3 steps:
- Made a list of all the main source source directories for each project, `SOURCE_DIR_LST`
- Ran `./scripts/get_dependendent_packages.sh SOURCE_DIR_LST` to get the package.txt for each source directory on each line of `SOURCE_DIR_LST`
- Ran `./scripts/get_potential_modules.sh SOURCE_DIR_LST OUTPUT_MODULE_LST` to get the list of all potential modules. 
