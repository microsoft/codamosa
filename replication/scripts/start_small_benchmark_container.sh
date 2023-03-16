#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
SCRIPT_PARENT_DIR=$(dirname "$SCRIPT_DIR")
mkdir ${SCRIPT_PARENT_DIR}/test-apps
docker run -v ${SCRIPT_PARENT_DIR}/test-apps:/home/codamosa/test-apps -it --name codamosa-benchmarks-container benchmarks-image  /bin/bash /home/codamosa/scripts/setup_only_necessary_test_apps.sh
