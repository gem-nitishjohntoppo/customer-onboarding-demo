FROM public.ecr.aws/docker/library/python:3.10-slim-bullseye

WORKDIR /app

# COPY pip.conf /etc/

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt


COPY . .

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
