FROM python:3.4-onbuild

RUN apt-get update && \
  apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran && \
  pip3 install scipy

COPY . /usr/src/tfmodels
WORKDIR /usr/src/tfmodels

CMD ["python"]