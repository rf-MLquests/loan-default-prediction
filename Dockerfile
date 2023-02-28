FROM jupyter/datascience-notebook

RUN mkdir loan-default-prediction

COPY . loan-default-prediction/

WORKDIR loan-default-prediction

RUN pip3 install -r requirements.txt

EXPOSE 9000