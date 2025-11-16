from app.api.main import app, configure_supervisor
from app.models.local_llm_client import LocalLLMClient

if __name__ == "__main__":
    client = LocalLLMClient(
        model_path="",               # ignored because server_url is set
        server_url="http://localhost:8001",
        temperature=0.2,
        max_tokens=512,
    )
    configure_supervisor(actor_client=client)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

