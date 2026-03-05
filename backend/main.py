from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routers import auth, cognitive, users, websocket

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Cognitive Load Estimator API",
    description="API for estimating student cognitive load using typing and mouse patterns",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(cognitive.router)
app.include_router(users.router)
app.include_router(websocket.router)


@app.get("/")
def root():
    return {"message": "Cognitive Load Estimator API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}
