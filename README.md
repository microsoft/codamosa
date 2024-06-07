# Codamosa

This repository contains the code for CodaMOSA. CodaMOSA integrates queries to a Large Language Model (currently supports the OpenAI API) into search-based algorithms for unit test generation. The paper on CodaMOSA will be published at ICSE'23:

> Caroline Lemieux, Jeevana Priya Inala, Shuvendu K. Lahiri, Siddhartha Sen. 2023. CODAMOSA: Escaping Coverage Plateaus in Test
Generation with Pre-trained Large Language Models. In *Proceedings of the 45th International Conference on Software Engineering*.

**CodaMOSA is implemented on top of the [Pynguin](https://github.com/se2p/pynguin) platform for Python unit test generation; this code base contains the Pynguin code as well as the CodaMOSA algorithm. If you would like to use or build on top of the Pynguin unit test generation part of CodaMOSA, consider building directly off of Pynguin; it is more frequently maintained than CodaMOSA.**

The main files added to implement CodaMOSA are:
```
- pynguin/generation/algorithms:
   - codamosastrategy.py: the modified version of mosastrategy.py that implements (1) tracking of coverage plateaus and (2) invoking Codex to generate new testcases. 
- pynguin/languagemodels: 
   - astscoping.py: defines a modified python AST that contains Pynguin VariableReferences in place of variable names, used to support uninterpreted statements in CodaMOSA. 
   - functionplaceholderadder.py: no longer used by CodaMOSA, allows to randomly mutate a given function with a placeholder "??"
   - model.py: the main interface to the Codex API
   - output_fixers.py: the AST rewriter used to normalize Codex-generated code to a format closer to Pynguin's output
```

## License

CodaMOSA buils on Pynguin 0.19.0, which was licensed LGPL-3.0. A copy of this license is available in the [LICENSES directory](LICENSES/LGPL-3.0-or-later.txt). Thefiles modified/added by CodaMOSA are licensed under the MIT license. A copy of this license is available in the [LICENSE.txt](LICSENSE). document. File headers outline the license under which each file is distributed.

Versions of Pynguin 0.30.0 and onwards are now licensed as MIT.

## Running

To run on a project, first build the runner:
```
docker build -t codamosa-runner -f docker/Dockerfile --platform linux/amd64 .
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
Note (June 2024): the model `code-davinci-002` is no longer available, you will need to update the command with a model you have access to. 

## Replication package

For information about replicating the results in the ICSE'23 submission, follow the instructions in the `replication` folder. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Building for Development

This project uses the `[poetry](https://python-poetry.org/)` dependency management program, and requires Python 3.10 to run. After you have poetry installed, you can navigate to the `codamosa` directory and edit the code following these steps:

1. First, create a virtual environment and install dependencies using:
```
$ poetry install
```
After the initial install, you can activate the virtual environment via:
```
$ poetry shell
```
2. After you have made your changes, you can run the linters and tests with the command:
```
$ make check
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
