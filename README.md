# Codamosa

This repository contains the code for CodaMOSA. 

CodaMOSA is implemented as a search algorithm on top of the [Pynguin](https://github.com/se2p/pynguin) platform for Python unit test generation; this code base contains the Pynguin code as well as the CodaMOSA algorithm. For general information about pynguin, refer to `Pynguin-README.md` in this repository. Like Pynguin, CodaMOSA is distributed under the GPL-3 license. 

The main files added/modified to implement CodaMOSA are:
```
- pynguin/generation/algorithms:
   - codamosastrategy.py: the modified version of mosastrategy.py that implements (1) tracking of coverage plateaus and (2) invoking Codex to generate new testcases. 
- pynguin/languagemodels: 
   - astscoping.py: defines a modified python AST that contains Pynguin VariableReferences in place of variable names, used to support uninterpreted statements in CodaMOSA. 
   - functionplaceholderadder.py: no longer used by CodaMOSA, allows to randomly mutate a given function with a placeholder "??"
   - model.py: the main interface to the Codex API
   - output_fixers.py: the AST rewriter used to normalize Codex-generated code to a format closer to Pynguin's output
```

## Running

To run on a project, first build the runner:
```
docker build -t codamosa-runner -f docker/Dockerfile --platform linux/amd64
```
(or use the target platform of your choice --- should match the machine on which you want to run experiments.)


You can then run 
```
docker run --rm -v TARGET_PROJECT_DIRECTORY:/input:ro -v OUTPUT_DIRECTORY:/output -v TARGET_PROJECT_DIRECTORY:/package:ro codamosa-runner <ARGS_TO_PYNGUIN> 
```
there must be a file called package.txt in `TARGET_PROJECT_DIRECTORY` which contains all the requirements that need to be installed for the target project in the `requirements.txt` format. The `pipreqs` tool can help you generate one automatically. 

Here is an example run, given that the `flutils` project is cloned under `$TEST_BASE/test-apps`, and that there is a `package.txt` file in `test-apps/flutils`:
```
$ mkdir /tmp/flutils-out
$ docker run --rm -v $TEST_BASE/test-apps/flutils:/input:ro -v /tmp/flutils-out:/output \
    -v $TEST_BASE/test-apps/flutils:/package:ro codamosa-runner --project_path /input \
    --module-name flutils.packages --output-path /output  --report-dir /output --maximum_search_time 120 \
    --output_variables TargetModule,CoverageTimeline --coverage_metrics BRANCH,LINE  --assertion-generation NONE \
    --algorithm CODAMOSA -v --include-partially-parsable True --allow-expandable-cluster True \
    --uninterpreted_statements ONLY --temperature 0.8 --model_name code-davinci-002 --authorization-key $AUTH_KEY"
```

## Replication package

For information about replicating the results in the ICSE'23 submission, follow the instructions in the supplementary material.

