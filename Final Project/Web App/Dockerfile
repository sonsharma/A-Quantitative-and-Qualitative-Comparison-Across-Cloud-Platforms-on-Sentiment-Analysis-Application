# Start with a Linux micro-container to keep the image tiny
FROM python:3.6

# Document who is responsible for this image
MAINTAINER Sonal Sharma "ss13449@nyu.edu"
MAINTAINER Kefei Wu "kw2669@nyu.edu"


# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD /app/requirements.txt /app
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Expose any ports the app is expecting in the environment
ENV PORT 8081
EXPOSE $PORT

# Add the code as the last Docker layer because it changes the most
ADD /app/app.py  /app/app.py
ADD /app/tokenizer_model.pkl  /app/tokenizer_model.pkl
ADD /app/templates/home.html  /app/templates/home.html

# Run the service
CMD [ "python", "app.py"]

