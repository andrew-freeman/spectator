from app.api.main import app, configure_supervisor
from app.models.openai_client import OpenAIClient

if __name__ == "__main__":
    client = OpenAIClient(model="gpt-4o-mini", api_key="TBD: INSERT YOUR API KEY HERE")
    configure_supervisor(actor_client=client)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

