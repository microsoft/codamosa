# Codamosa

This repository contains the code for codamosa. 

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


```
CMD="docker run --rm -v ${TEST_DIR}:/input:ro -v ${OUT_DIR}:/output -v ${TEST_DIR}:/package:ro pynguin-runner-local --assertion-generation NONE --project_path /input --module-name ${TEST_MOD} --output-path /output --maximum_search_time $SEARCH_TIME --output_variables TargetModule,Coverage,BranchCoverage,LineCoverage,ParsedStatements,UninterpStatements,ParsableStatements,LLMCalls,LLMQueryTime,LLMStageSavedTests,LLMStageSavedMutants,LLMNeededExpansion,LLMNeededUninterpreted,LLMNeededUninterpretedCallsOnly,RandomSeed,AccessibleObjectsUnderTest,CodeObjects,CoverageTimeline --report-dir /output --coverage_metrics BRANCH,LINE $ARGS --authorization-key $AUTH_KEY -v"
```


For general information about pynguin, refer to Pynguin-README.md in this repository.

