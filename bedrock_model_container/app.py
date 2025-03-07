from fastapi import FastAPI, Request
import boto3
import json

app = FastAPI()

@app.post("/invoke-llm")
async def invoke_llm(request: Request):
    req = await request.json()
    prompt = req.get("prompt")
    text = req.get("text")
    # Interact with LLM (example for AWS Bedrock)
    bedrock = boto3.client("bedrock-runtime")
    payload ={
        "prompt": f"Prompt: {prompt}\nText: {text}",
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
        
    }
    body = json.dumps(payload)
    response = bedrock.invoke_model(
        modelId="meta.llama3-3-70b-instruct-v1:0",  # Replace with your model ID
        contentType="application/json",
        accept="application/json",
        body=body
    )

    result = json.loads(response["body"].read().decode("utf-8"))
    return result
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)