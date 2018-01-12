FROM ubuntu:latest
MAINTAINER Andrea Pierleoni "andreap@ebi.ac.uk"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
#ENTRYPOINT ["python"]
EXPOSE 8080
CMD ["gunicorn", "-b", ":8080", "main:app"]