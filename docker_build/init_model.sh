
#! /bin/bash
torch-model-archiver --model-name dr-score \
--version 1.0 --model-file /home/model-server/model.py \
--serialized-file best.pth \
--handler dr_handler.py  \
--requirements-file requirements.txt \
--extra-files /home/model-server/handler.py

ls *.mar

mkdir model_store
mv dr-score.mar model_store