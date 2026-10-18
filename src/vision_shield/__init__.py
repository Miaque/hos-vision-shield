def main() -> None:
    import uvicorn

    uvicorn.run("vision_shield.main:app", host="127.0.0.1", port=8000, access_log=False, limit_concurrency=32)
