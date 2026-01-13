from fastapi import APIRouter
from .endpoints import auth, books, recommendations, ratings

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(books.router, prefix="/books", tags=["books"])
router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
router.include_router(ratings.router, prefix="/ratings", tags=["ratings"])


