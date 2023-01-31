#!/bin/bash
# This file lets you run one instance of codamosa, etc.  

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

if [ "$#" -lt 5 ]; then
	echo "Usage: $0 module-name out-dir argfile authkeyfile searchtime"
	exit 1
fi
MOD=$1
OUT_DIR=$2
ARGFILE=$3
AUTH_KEY=`cat $4`
SEARCH_TIME=$5
ARGS=`cat $ARGFILE`

if [[ ! -d $SCRIPT_DIR/../test-apps ]]; then
	echo "It seems that the directory test-apps does not exist. run ./start_benchmark_container.sh first."
	exit 1
fi

grep $1 $SCRIPT_DIR/../test-apps/good_modules.csv | 
while IFS=, read  -r TEST_DIR TEST_MOD
do
	TEST_DIR=$(dirname $SCRIPT_DIR)/$TEST_DIR 
	mkdir -p $OUT_DIR
CMD="docker run --rm -v ${TEST_DIR}:/input:ro -v ${OUT_DIR}:/output -v ${TEST_DIR}:/package:ro pynguin-runner --assertion-generation NONE --project_path /input --module-name ${TEST_MOD} --output-path /output --maximum_search_time $SEARCH_TIME --output_variables TargetModule,Coverage,BranchCoverage,LineCoverage,ParsedStatements,UninterpStatements,ParsableStatements,LLMCalls,LLMQueryTime,LLMStageSavedTests,LLMStageSavedMutants,LLMNeededExpansion,LLMNeededUninterpreted,LLMNeededUninterpretedCallsOnly,RandomSeed,AccessibleObjectsUnderTest,CodeObjects,CoverageTimeline --report-dir /output --coverage_metrics BRANCH,LINE $ARGS --authorization-key $AUTH_KEY -v"
	$CMD
	cat $OUT_DIR/statistics.csv
done
