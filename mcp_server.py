import os
import uuid
import shutil
import httpx
from typing import Any, Dict, List, Union, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64 

from shared_models import (
    ProcessRequest, ProcessResponse,
    IngestRequest, SearchRequest, SearchResponse,
    MCPResponse, Output, Reference,
    ToolDescriptor, ResourceDescriptor, PromptTemplate,
    ListToolsResponse, ListResourcesResponse, ListPromptsResponse,
    UpdatePromptRequest, UpdatePromptResponse
)

# ---- JSON-RPC 2.0 Envelope Models ----
class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: Union[Dict[str, Any], List[Any]]
    id: Union[int, str, None]

class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    result: Any
    id: Union[int, str, None]

class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Any = None

class JSONRPCErrorResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    error: JSONRPCError
    id: Union[int, str, None]

# ---- Global Prompt Templates ----
prompt_templates: Dict[str, str] = {
    "default": "\n".join([
        "You are a knowledgeable document assistant.",
        "If the user's question requires a figure or table to answer,",
        "  - Identify exactly one figure or one table (e.g. 'Figure 2' or 'Table 3'),",
        "  - Give its page number or location,",
        "  - Provide a description of what it shows,",
        "  - Then answer the user's question by referring to that figure/table.",
        "If the question does not need a figure or table, answer using text context only.",
        "Do not repeat the question in your answer or repeat the answers.",
        "Always respond in clear, structured Traditional Chinese."
    ])
}

# ---- Configuration ----
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PDF_DIR     = os.path.join(BASE_DIR, "pdfs")
THUMBS_DIR  = os.path.join(PDF_DIR, "thumbs")
IMAGES_DIR  = os.path.join(PDF_DIR, "images")
for d in (PDF_DIR, THUMBS_DIR, IMAGES_DIR):
    os.makedirs(d, exist_ok=True)

PDF_TOOL_URL = os.getenv('PDF_TOOL_URL', 'http://localhost:8001')
RAG_TOOL_URL = os.getenv('RAG_TOOL_URL',   'http://localhost:8002')
HOST_URL     = os.getenv('HOST_URL',       'http://localhost:8000')

app = FastAPI(
    title="MCP Server (JSON-RPC)",
    openapi_url=None,
    docs_url=None,
    redoc_url=None
)

# ---- Static & Frontend ----
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    with open(os.path.join(BASE_DIR, "index.html"), 'r', encoding='utf-8') as f:
        return HTMLResponse(f.read())

app.mount("/thumbs", StaticFiles(directory=THUMBS_DIR), name="thumbs")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ---- JSON-RPC Single Endpoint ----
@app.post("/mcp/v1/rpc")
async def rpc_endpoint(req: JSONRPCRequest):
    id_ = req.id
    try:
        # Dynamic discovery methods
        if req.method == "listTools":
            tools = [
                ToolDescriptor(
                    name="uploadPdf",
                    description="Upload base64 PDF and ingest",
                    params_schema={
                        "pdf_data": {"type":"string"},
                        "filename": {"type":"string"}
                    }
                ),
                ToolDescriptor(
                    name="processPdf",
                    description="Process PDF to MD, images, tables",
                    params_schema={"pdf_path": {"type": "string"}}
                ),
                ToolDescriptor(
                    name="ingest",
                    description="Ingest markdown into vector store",
                    params_schema={
                        "md_path": {"type":"string"},
                        "images": {"type":"object"},
                        "tables": {"type":"object"},
                        "source_filename": {"type":"string"}
                    }
                ),
                ToolDescriptor(
                    name="search",
                    description="Search context from vector store",
                    params_schema={
                        "question": {"type":"string"},
                        "k": {"type":"integer"},
                        "file_ids": {"type":"array"}
                    }
                ),
                ToolDescriptor(
                    name="query",
                    description="Model inference with text and images",
                    params_schema={"question": {"type":"string"}, "image_urls": {"type":"array"}}
                ),
                ToolDescriptor(
                    name="predict",
                    description="Full pipeline: process, ingest, search, query",
                    params_schema={"file_ids": {"type":"array"}, "question": {"type":"string"}, "k": {"type":"integer"}}
                ),
                ToolDescriptor(name="listResources", description="List available resources", params_schema={}),
                ToolDescriptor(name="listPrompts",   description="List prompt templates",  params_schema={}),
                ToolDescriptor(name="updatePrompt",  description="Update a prompt template", params_schema={"name": {"type":"string"}, "template": {"type":"string"}}),
                ToolDescriptor(
                    name="deleteSource",
                    description="Delete an uploaded PDF and all derived artifacts and its Milvus vectors",
                    params_schema={"source": {"type":"string"}}
                ),
            ]
            return JSONRPCResponse(result=ListToolsResponse(tools=tools).dict(), id=id_)

        if req.method == "listResources":
            resources = [
                ResourceDescriptor(name="thumbs", type="static", description="Thumbnail images", url="/thumbs"),
                ResourceDescriptor(name="images", type="static", description="Extracted images", url="/images"),
                ResourceDescriptor(name="pdfs",   type="dynamic", description="Uploaded PDFs", url="/pdfs")
            ]
            return JSONRPCResponse(result=ListResourcesResponse(resources=resources).dict(), id=id_)

        if req.method == "listPrompts":
            prompts = [PromptTemplate(name=n, template=t) for n, t in prompt_templates.items()]
            return JSONRPCResponse(result=ListPromptsResponse(prompts=prompts).dict(), id=id_)

        if req.method == "updatePrompt":
            up = UpdatePromptRequest(**req.params)
            prompt_templates[up.name] = up.template
            prompt = PromptTemplate(name=up.name, template=up.template)
            return JSONRPCResponse(result=UpdatePromptResponse(success=True, prompt=prompt).dict(), id=id_)
        # ----------  uploadPdf  ----------
        


        if req.method == "uploadPdf":
            b64 = req.params["pdf_data"]
            orig_name = req.params.get("filename") or f"{uuid.uuid4().hex}.pdf"
            bin_data  = base64.b64decode(b64)

            files = {"file": (orig_name, bin_data, "application/pdf")}
            async with httpx.AsyncClient(timeout=300) as cli:
                ing_res = await cli.post(f"{HOST_URL}/ingest", files=files)
            if ing_res.status_code != 201:
                raise HTTPException(502, f"host_app ingest error: {ing_res.text}")

        
            ingest_info = ing_res.json()        # {"status","source","chunks_added"}

            return JSONRPCResponse(result=ingest_info, id=id_)
        # ----------  uploadPdf  ----------
        # Functional methods
        if req.method == "processPdf":
            pr = ProcessRequest(**req.params)
            async with httpx.AsyncClient() as client:
                res = await client.post(f"{PDF_TOOL_URL}/process_pdf", json=pr.dict())
            data = ProcessResponse(**res.json())
            return JSONRPCResponse(result=data.dict(), id=id_)

        if req.method == "ingest":
            ir = IngestRequest(**req.params)
            async with httpx.AsyncClient() as client:
                res = await client.post(f"{RAG_TOOL_URL}/ingest", json=ir.dict())
            return JSONRPCResponse(result=res.json(), id=id_)

        if req.method == "search":
            sr = SearchRequest(**req.params)
            async with httpx.AsyncClient() as client:
                res = await client.post(f"{RAG_TOOL_URL}/search", json=sr.dict())
            data = SearchResponse(**res.json())
            return JSONRPCResponse(result=data.dict(), id=id_)

        if req.method == "query":
            async with httpx.AsyncClient() as client:
                res = await client.post(f"{HOST_URL}/query", json=req.params)
            return JSONRPCResponse(result=res.json(), id=id_)

        if req.method == "predict":
            params = req.params
            async with httpx.AsyncClient(timeout=300) as client:
                

                # search
                # edit
                sr = {'question': params['question'], 'k': params.get('k',15), "file_ids": params.get("file_ids")}
                # edit
                r2 = await client.post(f"{RAG_TOOL_URL}/search", json=sr)
                search = r2.json()
                # query
                thumbs = search.get('image_paths', [])
                urls = [f"{HOST_URL}/thumbs/{os.path.basename(p)}" for p in thumbs]
                #edit start
                # r3 = await client.post(f"{HOST_URL}/query", json={'question': params['question'], 'image_urls': urls})
                r3 = await client.post(
                    f"{HOST_URL}/query",
                    json={
                        'question':   params['question'],
                        'image_urls': urls,
                        'file_ids':   params.get('file_ids', [])
                    }
                )
                #edit end
                inf = r3.json()

            return JSONRPCResponse(
                result={
                    "answer": inf.get("answer", ""),
                    "sources": search.get("sources", [])
                },
                id=id_
            )
            #return JSONRPCResponse(result=out.dict(), id=id_)
        if req.method == "deleteSource":
            import glob
            src = os.path.basename(req.params["source"])          # 防止路徑穿越
            base, _ = os.path.splitext(src)

            removed = []

            # 1) 刪原始 PDF 與 Markdown
            for p in [os.path.join(PDF_DIR, src), os.path.join(PDF_DIR, base + ".md")]:
                if os.path.exists(p):
                    try:
                        os.remove(p); removed.append(p)
                    except Exception as e:
                        removed.append(f"{p} [ERR: {e}]")

            # 2) 刪 images / thumbs / debug 派生物
            for folder, pat in [
                (IMAGES_DIR, f"{base}_page*"),   # e.g. 2023_page1_img0.png
                (THUMBS_DIR, f"{base}_page*"),   # e.g. 2023_page1_img0_thumb.jpg
                (os.path.join(PDF_DIR, "debug"), f"{base}_page*"),
            ]:
                try:
                    for fp in glob.glob(os.path.join(folder, pat)):
                        try:
                            os.remove(fp); removed.append(fp)
                        except Exception as e:
                            removed.append(f"{fp} [ERR: {e}]")
                except Exception:
                    pass  # 目錄不存在就跳過

            # 3) 刪 Milvus 內該檔所有 chunks
            async with httpx.AsyncClient(timeout=60) as cli:
                rag_del = await cli.post(f"{RAG_TOOL_URL}/delete_source",
                                         json={"source_filename": src})
                rag_info = rag_del.json() if rag_del.headers.get("content-type","").startswith("application/json") else {}

            return JSONRPCResponse(result={
                "status": "ok",
                "removed": removed,
                "milvus": rag_info
            }, id=id_)

        # Method not found
        err = JSONRPCError(code=-32601, message="Method not found")
        return JSONRPCErrorResponse(jsonrpc="2.0", error=err, id=id_)

    except Exception as e:
        # JSON-RPC internal error
        err = JSONRPCError(code=-32603, message="Internal error", data=str(e))
        return JSONRPCErrorResponse(jsonrpc="2.0", error=err, id=id_)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8003, reload=True)
