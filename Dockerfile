# Use the official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Metaflow and other dependencies
RUN pip install metaflow wordcloud gensim pandas scikit-learn

COPY myflow.py .
COPY data_processing.py .
COPY open_ave_data.xlsx .
ENV USERNAME sample

# Run the Metaflow pipeline when the container starts
# Note: Metaflow's `run` command is used to execute the flow
CMD ["python", "myflow.py", "--df", "open_ave_data.xlsx"]
