FROM pytorch/torchserve:0.1-cpu


# Copy files over
COPY docker_build/init_model.sh /home/model-server/init_model.sh
COPY docker_build/scripts/model.py /home/model-server/model.py
COPY docker_build/scripts/handler.py /home/model-server/handler.py
COPY docker_build/scripts/dr_handler.py /home/model-server/dr_handler.py
COPY docker_build/model/best.pth /home/model-server/best.pth
COPY docker_build/requirements.txt /home/model-server/requirements.txt

# COPY docker_build/download_model.sh /home/model-server/download_model.sh
# Prepare to start server
USER root
# RUN chmod +x /home/model-server/download_model.sh
RUN chmod +x /home/model-server/init_model.sh
EXPOSE 8080 8081
RUN pip install torch-model-archiver -q
USER model-server
# For creating models
# RUN /home/model-server/download_model.sh
RUN /home/model-server/init_model.sh
