FROM python:3.12

# albumentations needs opencv-python-headless which broke on 3.13
# https://github.com/opencv/opencv-python/issues/993
# :)

LABEL version="0.1.0"
LABEL description="Docker image that opens Jupyter Lab, prepared to do data preparation and analysis."

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
VOLUME ["/data"]

ENTRYPOINT ["sh", "run.sh"]
