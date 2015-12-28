FROM b.gcr.io/tensorflow/tensorflow

# Pull latest branch from github
RUN git clone https://github.com/dennybritz/tf-models.git

WORKDIR tf-models