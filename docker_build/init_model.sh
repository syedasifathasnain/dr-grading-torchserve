
torch-model-archiver --model-name dr-score \
--version 1.0 --model-file /home/model-server/model.py \
--serialized-file best.pth \
--handler dr_handler.py  \
--requirements-file requirements.txt \
--extra-files /home/model-server/handler.py

ls *.mar

mkdir model_store
mv dr-score.mar model_store/

#response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar&initial_workers=1&synchronous=true")
#
#if [ ! "$response" == 200 ]
#then
#    echo "failed to register model with torchserve"
#else
#    echo "successfully registered resnet-18 model with torchserve"
#fi
#
#echo "TorchServe is up and running with resnet-18 model"
#echo "Management APIs are accessible on http://127.0.0.1:8081"
#echo "Inference APIs are accessible on http://127.0.0.1:8080"
#echo "For more details refer TorchServe documentation"
#echo "To stop docker container for TorchServe use command : docker container stop $container_id"