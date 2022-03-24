FROM pytorch/torchserve:latest

# Work directory
WORKDIR /home/model-server

# Copy files over
COPY docker_build /home/model-server/docker_build

# Prepare to start server

# USER root
# RUN chmod +x /home/model-server/download_model.sh
# RUN chmod +x /home/model-server/docker_build/init_model.sh

EXPOSE 8080 9000
RUN pip install torch-model-archiver -q
USER model-server

# For creating models
# RUN /home/model-server/download_model.sh
# RUN bash /home/model-server/init_model.sh

RUN torch-model-archiver --model-name dr-score --version 1.0 --model-file docker_build/scripts/model.py --serialized-file docker_build/model/best.pth --handler docker_build/scripts/dr_handler.py  --requirements-file docker_build/requirements.txt --extra-files docker_build/scripts/handler.py
RUN mkdir model_store
RUN mv dr-score.mar model_store

CMD [ "torchserve", "--start" , "--model-store model_store", "--models dr-score=dr-score.mar" ]
