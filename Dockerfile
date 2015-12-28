FROM python:3.4-onbuild

COPY . /usr/src/tfmodels
WORKDIR /usr/src/tfmodels

CMD ["python"]