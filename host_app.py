import os
import shutil
import httpx 
import torch
import logging 
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
#from fastapi.templating import Jinja2Templates
# 0914 edit---------------------------------------------------------------
# from transformers import (
#     Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# )
##from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
#
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
#--------------------------------------------------------------------------
from qwen_vl_utils import process_vision_info
from shared_models import ProcessRequest, IngestRequest, SearchRequest

# --- logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Setup ---
PDF_TOOL_URL = os.getenv("PDF_TOOL_URL", "http://localhost:8001")
RAG_TOOL_URL = os.getenv("RAG_TOOL_URL", "http://localhost:8002")
PDF_DIR = "./pdfs"
IMG_DIR = os.path.join(PDF_DIR, "images")
THUMB_DIR = os.path.join(PDF_DIR, "thumbs")

app = FastAPI()

# --- Load AI Model ---
logging.info("Loading AI model...")
# --- Default Prompt Templates (fallback) ---

#--------load prompts from MCP server--------
async def load_prompts():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8003/mcp/v1/rpc",
            json={"jsonrpc":"2.0","method":"listPrompts","params":{},"id":1}
        )
    data = resp.json()["result"]["prompts"]
    return [p["template"] for p in data]
#-----------------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# 0914 edit------------------------------------------------------------------
##processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
##tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
# )
##config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
#
# # model = AutoModelForCausalLM.from_pretrained(
# #     MODEL_NAME,
# #     torch_dtype=torch.float16,
# #     device_map="auto",
# #     trust_remote_code=True,
# # )
####config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    ####config=config,
    dtype=torch.float16,      
    device_map="auto",
    trust_remote_code=True,
)

#---------------------------------------------------------------------------
model.eval()
logging.info("AI model loaded successfully.")


@app.post('/ingest')
async def ingest_pipeline(file: UploadFile = File(...)):
    # ... (ingest)
    pdf_path = os.path.join(PDF_DIR, file.filename)
    with open(pdf_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    async with httpx.AsyncClient(timeout=300.0) as client:
        process_req = ProcessRequest(pdf_path=pdf_path)
        pdf_res = await client.post(f"{PDF_TOOL_URL}/process_pdf", json=process_req.dict())
        if pdf_res.status_code != 200:
            return JSONResponse(status_code=pdf_res.status_code, content=pdf_res.json())
        processed_data = pdf_res.json()

        ingest_req = IngestRequest(
            md_path=processed_data['md_path'],
            images=processed_data['images'],
            tables=processed_data['tables'],
            source_filename=file.filename
        )
        rag_res = await client.post(f"{RAG_TOOL_URL}/ingest", json=ingest_req.dict())

    return JSONResponse(content=rag_res.json(), status_code=rag_res.status_code)

@app.post('/query')
async def query_pipeline(request: Request):
    try:
        logging.info("--- Query request received ---")
        torch.cuda.empty_cache()
        req_json = await request.json()
        question = req_json.get("question")
        #add start
        file_ids = req_json.get("file_ids", [])
        #add end

        prompt_templates = req_json.get("prompt_templates")
        if not prompt_templates:
            prompt_templates = await load_prompts()
        logging.info(f" Prompt Templates：{prompt_templates}")

        print(prompt_templates)
        #-----------------------------------------------
        logging.info(f"Question: {question}")

        logging.info("Step 1: Calling RAG tool server for search...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            #edit start
            search_payload = {"question": question, "k": 15}
            if file_ids:
             search_payload["file_ids"] = file_ids
            rag_res = await client.post(f"{RAG_TOOL_URL}/search", json=search_payload)
            #edit end
            if rag_res.status_code != 200:
                logging.error(f"RAG tool server returned error {rag_res.status_code}: {rag_res.text}")
                return JSONResponse(status_code=rag_res.status_code, content=rag_res.json())
            retrieved_data = rag_res.json()
        logging.info(f"Step 1 successful. Retrieved {len(retrieved_data.get('sources', []))} sources.")

        logging.info("Step 2: Building prompt for AI model...")
        context = retrieved_data['context']
        thumb_urls = [f"http://localhost:8003/thumbs/{os.path.basename(p)}" for p in retrieved_data['image_paths']]
        print(thumb_urls)
        prompt_text = "\n\n".join([
            *prompt_templates,
            f"Context:\n{context}",
            f"Question:\n{question}"
        ])
        
        conv = [{'role':'user', 'content': [{'type':'text', 'text': prompt_text}, *[{'type':'image', 'image': u} for u in thumb_urls]]}]
        logging.info("Step 2 successful. Prompt built.")

        logging.info("Step 3: Processing inputs for model...")
        text_ = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        from fastapi.concurrency import run_in_threadpool
        img_in, vid_in = await run_in_threadpool(process_vision_info, conv)
        #img_in, vid_in = process_vision_info(conv)
        inputs = processor(text=[text_], images=img_in, videos=vid_in, padding=True, return_tensors='pt')
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        logging.info("Step 3 successful. Inputs processed and moved to device.")

        logging.info("Step 4: Starting AI model generation...")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=1024)
        logging.info("Step 4 successful. AI model generation complete.")

        logging.info("Step 5: Decoding response...")

        seq = inputs['input_ids'].shape[1]
        ans = processor.batch_decode([gen[0][seq:]], skip_special_tokens=True)[0]
        logging.info("Step 5 successful. Response decoded.")

        return JSONResponse({'answer': ans, 'sources': retrieved_data['sources']})

    except Exception as e:
        logging.error(f"An unexpected error occurred in query_pipeline: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred in the host app."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)