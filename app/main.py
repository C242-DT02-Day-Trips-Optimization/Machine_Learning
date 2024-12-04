from fastapi import FastAPI
from app.routes import clustering_router

# Initialize FastAPI app
app = FastAPI()

# Include the clustering router
app.include_router(clustering_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cluster API!"}

if __name__ == "__main__":
    import uvicorn

    # Gunakan PORT dari environment atau default ke 8080
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)