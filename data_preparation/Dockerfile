FROM python:3.13-alpine

LABEL version="0.1.0"
LABEL description="Docker image that opens Jupyter Lab, prepared to do data preparation and analysis."

WORKDIR /app

COPY . .

RUN apk add gcc python3-dev musl-dev linux-headers

RUN pip install -r requirements.txt

EXPOSE 8888
VOLUME ["/data"]

ENTRYPOINT ["sh", "run.sh"]
