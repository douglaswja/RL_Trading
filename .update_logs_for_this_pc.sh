#!/bin/bash

# IMPORTANT NOTE: MOVE TO YOUR `mlruns` DIRECTORY AND PLACE THIS FILE THERE
# YOU SHOULD BE IN THE `mlruns` DIRECTORY WHEN YOU RUN THIS SCRIPT

# Execute this file via:
# ./update_logs_for_this_pc.sh
# 
# This bash script updates MLFlow's meta.yaml files.
# The yaml files contain artifact-directory keys with values to artifact-directory paths.
# However, these paths are unique to each computer, requiring them to be updated when these logs 
# are passed around.


FILES=$(find "$(cd ..; pwd)" -name "meta.yaml")

for FILE in $FILES
do
	if [[ $FILE =~ /[0-9]+/meta\.yaml ]]; then
		REPLACEMENT_TEXT=$(echo "artifact_location: file://$FILE" | sed "s/\/meta.yaml//g")
		echo $REPLACEMENT_TEXT > $FILE.tmp
		tail -n +2 "$FILE" >> "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
	else
		REPLACEMENT_TEXT=$(echo "artifact_uri: file://$FILE" | sed "s/meta.yaml/artifacts/g")
		echo $REPLACEMENT_TEXT > $FILE.tmp
		tail -n +2 "$FILE" >> "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
	fi
done
