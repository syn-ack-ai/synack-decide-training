#!/bin/sh
# Wait for v7 labelling (Mac client), build the student mix, copy to dreamer, start training there (detached).
cd ~/systemone
until grep -q LABEL_DONE "$1"; do sleep 60; done
PYTHONPATH=v2 python3 v2/build_student.py
ssh dreamer 'mkdir -p ~/synack/student'
scp -q v2/data/student/train.jsonl v2/data/student/valid.jsonl v2/train_hf.py jobs/run_student_dreamer.sh dreamer:~/synack/student/
scp -q bench/labels255.json dreamer:~/synack/student/
ssh dreamer 'cd ~/synack/student && chmod +x run_student_dreamer.sh && (setsid nohup ./run_student_dreamer.sh > train.log 2>&1 < /dev/null &) && echo started'
echo CHAIN_STARTED_TRAINING
