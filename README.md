# RecSys

This project is a Python application that uses collaborative filtering to recommend movies to users. It supports both local and containerized environments for easy setup and deployment.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Optional: [Conda](https://docs.conda.io/en/latest/) for a local environment setup.

---

## Project Structure

The project directory should look like this:

```text
/project-root
├── Dockerfile
├── docker-compose.yml
├── setup.py
├── requirements.txt       # Contains the dependencies
├── src/                   # Your Python code here
├── data/                  # Your data here
└── setup/                 # Environment setup files
```

---

## Installation

You have two options for setting up the environment:

### Option 1: Using Conda (Simpler)

This method is ideal for local development if you are not familiar with Docker.

1. **Navigate to the repository root directory**:
   ```bash
   cd recommender
   ```

2. **Create a Conda environment**:
   ```bash
   conda env create -f ./setup/movie-recommender.yml
   ```

3. **Activate the Conda environment**:
   ```bash
   conda activate movie-recommender
   ```

4. **Run the main program**:
   Execute the following command from the project root:
   ```bash
   python src/main.py
   ```

### Option 2: Using Docker (Recommended)

1. **Build the Docker image**:
   Run the following command to build the Docker image:
   ```bash
   docker-compose build
   ```

2. **Start the Docker container**:
   After building the image, start the container with:
   ```bash
   docker-compose up
   ```

   This will run the main program, and the output will be visible in the console.

---

## User-Based Movie Recommendation API

This FastAPI application provides an API for generating movie recommendations using a user-based collaborative filtering approach. It supports several key features:

### Features
- **Health Check**: Ensure the service is up and running.
- **Update Neighbors**: Recalculate and update user ratings and neighbors based on a dataset.
- **Generate Recommendations**: Provide personalized movie recommendations for a specific user.

---

## Running the API

You can run the API either directly with FastAPI or using Docker.

### Run Locally with FastAPI
1. **Ensure the prerequisites are met**:
   - Python 3.7 or later
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the application**:
   ```bash
   uvicorn --app-dir ./src app:app --host 0.0.0.0 --port 8989 --reload
   ```

### Run Using Docker
1. **Edit the `docker-compose.yml` file** as needed for your setup.

2. **Run the application with Docker**:
   ```bash
   docker-compose up
   ```

The application will be accessible at `http://localhost:8989`.

---

## Contributing

Contributions are welcome! If you have any suggestions, bug fixes, or new features to add:
1. Open an issue to discuss your ideas.
2. Submit a pull request with your changes.

Let’s make this project better together!

