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
ARGS=`cat $ARGFILE`
SEARCH_TIME=$4
PLAY_OPTION=$5 #"--auth" for querying codex with auth key or "--replay" with existing codex generations
PLAY_KEY=$6 # auth key file or codex generations file

if [[ ! -d $SCRIPT_DIR/../test-apps ]]; then
	echo "It seems that the directory test-apps does not exist. run ./start_benchmark_container.sh first."
	exit 1
fi

grep $1 $SCRIPT_DIR/../test-apps/good_modules.csv |
while IFS=, read  -r TEST_DIR TEST_MOD
do
	TEST_DIR=$(dirname $SCRIPT_DIR)/$TEST_DIR
	mkdir -p $OUT_DIR
	if [ $PLAY_OPTION = "--auth" ]; then
		PLAY_CONFIG="--authorization-key $(cat $PLAY_KEY)"
	elif [ $PLAY_OPTION = "--replay" ]; then
		PLAY_CONFIG="--replay-generation-from-file $PLAY_KEY"
	else
		echo "Invalid option $PLAY_OPTION"
		exit 1
	fi
CMD="docker run --rm -v ${TEST_DIR}:/input:ro -v ${OUT_DIR}:/output -v ${TEST_DIR}:/package:ro pynguin-runner --assertion-generation NONE --project_path /input --module-name ${TEST_MOD} --output-path /output --maximum_search_time $SEARCH_TIME --output_variables TargetModule,Coverage,BranchCoverage,LineCoverage,ParsedStatements,UninterpStatements,ParsableStatements,LLMCalls,LLMQueryTime,LLMStageSavedTests,LLMStageSavedMutants,LLMNeededExpansion,LLMNeededUninterpreted,LLMNeededUninterpretedCallsOnly,RandomSeed,AccessibleObjectsUnderTest,CodeObjects,CoverageTimeline --report-dir /output --coverage_metrics BRANCH,LINE $ARGS $PLAY_CONFIG -v"
	$CMD
	cat $OUT_DIR/statistics.csv
done
