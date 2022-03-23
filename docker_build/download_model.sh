echo "Registering densenet model"
curl -O https://download.pytorch.org/models/best.pth
curl -O https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/inception_v3/model.py
curl -O https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/dr_handler.py
curl -O https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/handler.py
curl -O https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/requirements.txt
