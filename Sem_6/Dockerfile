
##===== Initial distribution =====##
FROM python:3

##===== Installing packages and dependencies =====##
RUN apt-get update
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

##===== Creating folders =====##
WORKDIR workspace/

##===== Transfering files =====##
COPY hsd_descriptive.py hsd_descriptive.py 
COPY healthcare-dataset-stroke-data.csv healthcare-dataset-stroke-data.csv

