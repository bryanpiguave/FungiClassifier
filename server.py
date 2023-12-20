from typing import Union
from fastapi import FastAPI
import torch
import argparse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    default="export_model", help="Model file name")
parser.add_argument("--port", type=int, default=8000, help="Port number")
args = parser.parse_args()

app = FastAPI()


"""
This server receives an image and gives a prediction.
"""


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=args.port)
