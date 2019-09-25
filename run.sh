#!/bin/bash
rm output.txt
for i in `seq 1 50`;
do 
	echo "running ${i}"
	python2 capture.py -r braindead.py -b improvedTeam.py >> output.txt
done
grep -E "Blue team wins|Blue team has" output.txt | wc -l; grep -E "Red team wins|Red team has" output.txt | wc -l; grep "Tie" output.txt | wc -l
