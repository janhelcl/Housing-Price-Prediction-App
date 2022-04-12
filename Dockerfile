FROM python:3.9.4

WORKDIR /opt/rest_api

ARG PIP_EXTRA_INDEX_URL

ADD ./rest_api /opt/rest_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/rest_api/requirements.txt

RUN chmod +x /opt/rest_api/run.sh

EXPOSE 8001

CMD ["bash", "./run.sh"]