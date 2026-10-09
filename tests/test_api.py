"""API tests: the full upload -> ask -> delete flow with a fake LLM."""


def upload(client, name="lanterns.txt", content=b"Lanterns traditionally burn oil through a wick."):
    return client.post("/documents", files={"file": (name, content, "text/plain")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete(client):
    resp = upload(client)
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["filename"] == "lanterns.txt" and doc["chunks"] == 1

    assert client.get("/documents").json() == [doc]

    resp = client.post("/ask", json={"question": "What do lanterns burn?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Lanterns use wicks [1]."
    assert body["sources"][0] == {
        "index": 1,
        "document_id": doc["id"],
        "document": "lanterns.txt",
        "page": 1,
        "snippet": "Lanterns traditionally burn oil through a wick.",
    }

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get("/documents").json() == []


def test_upload_rejects_unsupported_type(client):
    resp = upload(client, name="data.csv", content=b"a,b")
    assert resp.status_code == 415


def test_upload_rejects_empty_file(client):
    resp = upload(client, name="empty.txt", content=b"   ")
    assert resp.status_code == 422


def test_upload_strips_client_path(client):
    resp = upload(client, name="../../etc/evil.txt")
    assert resp.status_code == 201
    assert resp.json()["filename"] == "evil.txt"


def test_delete_unknown_document_is_404(client):
    assert client.delete("/documents/nope").status_code == 404


def test_ask_with_no_documents_still_answers(client):
    resp = client.post("/ask", json={"question": "Anything?"})
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422