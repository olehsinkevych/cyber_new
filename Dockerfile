# pull official docker image
FROM python:3.10-slim-buster
LABEL maintainer='Oleh Sinkevych <oleh.sinkevych@lnu.edu.ua>'

# env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# set workdir
WORKDIR /cyber_app

COPY requirements.txt .
RUN /opt/venv/bin/python3 -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# copy project
COPY . /cyber_app

EXPOSE 8000

CMD ["python"]