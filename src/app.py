import logging
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse, RedirectResponse

# Your existing modules
from recommender.core import (
    PreProcessor,
    recommend_ub
)
from recommender.file_operations import (
    DataHandler,
    pickle_object,
    unpickle_object
)
from recommender.utils import setup_logging

# Setup logging
logging_path: Path = Path("loggers")
setup_logging(logger_root_dir=logging_path, log_level="DEBUG")

logger = logging.getLogger(__name__)
logger.info("FastAPI App starting...")

# FastAPI application instance
app = FastAPI(
    title="User-based Recommender API",
    description="This is a simple API that generates a list of recommended movies for a user.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# --- Global variables
data_handler = DataHandler(input_folder=Path("./data") / 'ml-latest-small')
movies_df: pd.DataFrame = data_handler.movies_df

try:
    user_ratings = unpickle_object(Path("./data/models/user_ratings.pkl"))
    neighbors_u = unpickle_object(Path("./data/models/user_neighbors.pkl"))
    logger.info("Pre-calculated ratings and neighbors loaded.")

except FileNotFoundError:
    logger.warning("Pre-calculated ratings and neighbors not found. Generating...")

    # Preprocess ratings and calculate user neighbors
    preprocessor = PreProcessor()
    user_ratings, neighbors_u = preprocessor.preprocess(data_handler.ratings_df)

    # Pickle the objects for future use
    logger.warning("Saving pre-calculated ratings and neighbors under: `./data/models`")
    pickle_object(user_ratings, Path("./data/models/user_ratings.pkl"))
    pickle_object(neighbors_u, Path("./data/models/user_neighbors.pkl"))


# Input validation classis
class RecommendationRequest(BaseModel):
    neighbors_num: int = 10
    recommendations_num: int = 5
    user_id: int = 100


class Movie(BaseModel):
    movieId: int
    title: str
    movie_genres: str
    recommendedScore: float


class DataPathRequest(BaseModel):
    data_path: str


# Health check endpoint
@app.get("/health")
async def health() -> JSONResponse:
    logger.info("Health check request received.")
    return JSONResponse({"status": "UP"})


# redirect to docs
@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


# Function to calculate user ratings and neighbors
def calculate_neighbors(data_path: str):
    global user_ratings, neighbors_u, movies_df
    logger.info(f"Calculating user ratings and neighbors from data at {data_path}...")

    data_handler = DataHandler(input_folder=Path("./data") / data_path)

    # Preprocess ratings and calculate user neighbors
    preprocessor = PreProcessor()
    user_ratings, neighbors_u = preprocessor.preprocess(data_handler.ratings_df)

    logger.info(f"Calculated user ratings and neighbors for {len(neighbors_u)} users.")

    # Update default data path
    movies_df = data_handler.movies_df


# Helper function to generate recommendations
def generate_recommendations(neighbors_num: int, recommendations_num: int, user_id: int) -> List[Movie]:
    global movies_df
    if user_ratings is None or neighbors_u is None or movies_df is None:
        raise ValueError("User ratings and neighbors must be calculated first.")

    logger.info(f"Generating recommendations for user {user_id}...")

    # Generate recommendations using the user-based collaborative filtering function
    recommended_movies = recommend_ub(
        user=user_id,
        movies_df=movies_df,
        neighbors_u=neighbors_u,
        user_ratings=user_ratings,
        neighbor_num=neighbors_num,
        rec_num=recommendations_num
    )

    # Return the list of movie titles as Movie models
    return [Movie(**movie) for movie in recommended_movies]


# FastAPI route for calculating neighbors (user ratings & neighbors_u)
@app.put("/api/v1/update_neighbors/")
async def update_neighbors_endpoint(request: DataPathRequest):
    try:
        calculate_neighbors(data_path=request.data_path)
        return JSONResponse(
            {"status": "SUCCESS",
             "message": "User ratings and neighbors have been calculated and stored."}
        )

    except Exception as e:
        logger.error(f"Error calculating neighbors: {str(e)}")
        return JSONResponse(
            {"status": "FAILURE",
             "message": f"Error calculating neighbors: {str(e)}"},
            status_code=500
        )


# FastAPI route for generating recommendations
@app.post("/api/v1/recommendations/", response_model=List[Movie])
async def generate_recommendations_endpoint(request: RecommendationRequest):
    try:
        # Generate recommendations using stored user ratings and neighbors_u
        recommended_movies = generate_recommendations(
            neighbors_num=request.neighbors_num,
            recommendations_num=request.recommendations_num,
            user_id=request.user_id
        )

        return recommended_movies

    except ValueError as e:
        logger.exception(f"Error: {str(e)}")
        return JSONResponse(
            {"status": "FAILURE",
             "message": {f"An error occurred while generating recommendations: {str(e)}"}},
            status_code=400
        )
    except Exception as e:

        logger.exception(f"Error generating recommendations: {str(e)}")
        return JSONResponse(
            {"status": "FAILURE",
             "message": {f"An error occurred while generating recommendations: {str(e)}"}},
            status_code=500
        )


# Main entry point for running the FastAPI app (if needed for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
