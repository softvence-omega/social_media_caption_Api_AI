

# from fastapi import FastAPI
# from app.api.v1.api import api_router

# app = FastAPI(root_path="/api/ai", title="Caption Generator")

# @api_router.get("/")
# def api_root():
#     return {"message": "API v1 working"}
# # Register API routes
# app.include_router(api_router)

import utils.transport
import utils.session
from fastapi import FastAPI
from app.api.v1.api import api_router

# 👇 Tell FastAPI it’s served under /api/ai
app = FastAPI(
    title="Caption Generator",
    root_path="/api/ai"
)

# ✅ Register all routes from api_router
app.include_router(api_router)
