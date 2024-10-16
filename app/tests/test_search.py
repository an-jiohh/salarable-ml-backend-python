from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "FastAPI is running"}

def test_query_search():
    response = client.post("/query_search", json={"keyword": "react", "sentence": "React is a frontend library."})
    assert response.status_code == 200
    assert isinstance(response.json(), list)