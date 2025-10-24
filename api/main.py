from fastapi import FastAPI


# DEFINE APP ENTRY

app = FastAPI()


# Get route

@app.get("/")
async def check_health():
    return "This is successful"


