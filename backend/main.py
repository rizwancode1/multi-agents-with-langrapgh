import os

import uvicorn

from app.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    if settings.is_production:
        host = "0.0.0.0"
        reload = False
        workers = int(os.getenv("WORKERS", 4))
    else:
        host = "127.0.0.1"
        reload = True
        workers = 1

    uvicorn.run(
        "main:app",
        host=host,
        port=8000,
        reload=reload,
        workers=workers,
    )
