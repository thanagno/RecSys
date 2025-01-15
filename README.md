# Recommender

This project is a Python application that uses collaborative filtering to recommend movies to users.
It uses Docker for easy setup and deployment.

## Prerequisites

Before you begin, ensure that you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Project Structure

The project directory should look like this:

```text
/project-root
├── Dockerfile
├── docker-compose.yml
├── setup.py
├── requirements.txt  # Contains the dependencies
├── src/           # Your Python code here
└── data/             # Your data here
```

## Installation

You have two options for setting up the environment:

### Option 1: Using Conda (Simpler)

This is the simplest way, in case you are not familiar with Docker. However, the recommended approach is through
docker.

1. **cd to the repository**:
   cd to the root directory of the downloaded data (suppose under a folder named `recommender`) cd

```bash
cd recommender
```

2. **Create a Conda environment**:

```bash
conda env create -f ./setup/movie-recommender.yml
```

3. **Activate the environment**:

```bash
conda activate movie-recommender
```

3. **Run the main function**:
   Once done execute the following command (from the project root):

```bash
python src/main.py
```

### Option 2: Using Docker

1. **Build the Docker Image:**
   Run the following command to build the Docker image:

```bash
docker-compose build
```

This will build the image according to the instructions in the Dockerfile.

2. **Start the Docker Container**:

Once the image is built, start the container with:

```bash
docker-compose up
```

This will run the main program, which you will be able to see in the output.

# User-based Movie Recommendation API

This FastAPI application serves as an API for generating movie recommendations using a user-based collaborative
filtering approach. The API allows you to request movie recommendations based on pre-calculated user ratings and
neighbor data.

## Features

- **Health Check**: Check if the service is up and running.
- **Update Neighbors**: Recalculate and update the user ratings and neighbors based on a provided dataset.
- **Generate Recommendations**: Get movie recommendations for a specific user based on their ratings and a user-based
  collaborative filtering model.

## Prerequisites

Before running the application, ensure you have the following:

- Python 3.7 or later
- Necessary Python packages (listed below)
- A dataset of movie ratings (default dataset: `ml-latest-small`)

## Development
You can run the application using uvicorn directly:
```bash
uvicorn --app-dir ./src app:app --host 0.0.0.0 --port 8989 --reload
```

The `docker-compose.yml` file can be edited 

# Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, bug fixes, or new
material to add.
#   R e c S y s  
 