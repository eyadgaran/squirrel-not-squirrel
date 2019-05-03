# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7

# Set an environment variable with the directory
# where we'll be running the app
ENV APP /app


# Create the directory and instruct Docker to operate
# from there from now on
RUN mkdir $APP
WORKDIR $APP


# Copy the requirements file in order to install
# Python dependencies
COPY requirements.txt .


# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt


# Copy local code to the container image.
# ADD . .
ADD main.py .
ADD secrets.conf .


# Run the web service on container startup
ENV SIMPLEML_CONFIGURATION_FILE ./secrets.conf
# GCP Overwrites this and exposes network ports as "Host"
ENV PORT 8081
EXPOSE $PORT
CMD hypercorn -w 1 main:app -k uvloop -b 0.0.0.0:${PORT}
