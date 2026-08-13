import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.vitis import store


@pytest.fixture()
def vitis_root(tmp_path):
    store.set_vitis_root(str(tmp_path / "vitis"))
    yield tmp_path / "vitis"
    store.set_vitis_root("runs/vitis")


@pytest.fixture()
def vitis_client(vitis_root):
    from sensorflow.vitis import api as vitis_api
    app = FastAPI()
    app.include_router(vitis_api.router)
    with TestClient(app) as client:
        yield client
