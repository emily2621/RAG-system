# rag_tool_server.py
import os
from fastapi import FastAPI, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus.vectorstores.milvus import Milvus as MilvusVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, utility, Collection
from pydantic import BaseModel
from shared_models import IngestRequest, SearchRequest, SearchResponse 

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION = "doc_chunks_mcp"

app = FastAPI()

# --- Milvus & Embeddings Setup ---
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
if utility.has_collection(COLLECTION):
    utility.drop_collection(COLLECTION)

embed_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
vector_store = MilvusVectorStore(
    embedding_function=embed_model,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    collection_name=COLLECTION,
    auto_id=True
)

# 0826 add 
class DeleteSourceRequest(BaseModel):
    source_filename: str  # 例: "202326803.pdf"
#

@app.post("/ingest", status_code=201)
async def ingest_document(req: IngestRequest):
    """Reads a processed markdown file and ingests it into the vector store."""
    try:
        if not os.path.exists(req.md_path):
            raise HTTPException(status_code=404, detail=f"Markdown file not found: {req.md_path}")

        with open(req.md_path, encoding='utf-8') as f:
            full_md = f.read()

        pages = full_md.split("<!-- Page ")[1:]  
        metas, texts = [], []
        for page_seg in pages:
            header, *body = page_seg.split("-->\n", 1)
            page_num = int(header.split(" ")[0])
            page_text = body[0] if body else ""
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
            page_chunks = splitter.split_text(page_text)
            
            # ==== 新增：若該頁沒有任何文字 chunk，但有圖片，就補上一個 sentinel chunk ====
            if (not page_chunks) and req.images.get(page_num):
                page_chunks = [f"[IMAGE-ONLY PAGE] {req.source_filename} page {page_num}"]
            # ===============================================================================


            for chunk_i, chunk_text in enumerate(page_chunks):
                texts.append(chunk_text)               
                metas.append({
                    "source": req.source_filename,
                    "page": page_num,
                    "chunk_in_page": chunk_i, 
                    "global_chunk": len(texts)-1,
                    "images": ";".join(req.images.get(page_num, [])), 
                    #"images": ";".join(req.images.get(page_num, [])[:1]), 
                })
            
        if not texts:
            return {"status": "ok", "source": req.source_filename, "chunks_added": 0, "message": "No text content to ingest."}
  
        vector_store.add_texts(texts, metas)
        return {"status": "ok", "source": req.source_filename, "chunks_added": len(texts)}

    except Exception as e:
        import traceback
        print(f"Ingestion Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_context(req: SearchRequest):
    """Searches for context relevant to the question (prefilter at vector DB)."""
    import math

    def build_expr(ids, use_json=False):
        # use_json=False -> 使用欄位名 `source`
        # use_json=True  -> 使用 JSON 欄位 `metadata["source"]`
        if not ids:
            return None
        if len(ids) == 1:
            fid = ids[0].replace('"', r'\"')
            return (f'source == "{fid}"' if not use_json
                    else f'metadata["source"] == "{fid}"')
        quoted = ", ".join([f'"{fid.replace(chr(34), r"\"")}"' for fid in ids])
        return (f'source in [{quoted}]' if not use_json
                else f'metadata["source"] in [{quoted}]')

    async def search_with_fallback(question, k, ids):
        """
        先嘗試以 `source` 欄位過濾；若欄位不存在則回退到 `metadata["source"]`。
        """
        # 1) try: source
        expr = build_expr(ids, use_json=False)
        try:
            return vector_store.similarity_search_with_score(question, k=k, expr=expr)
        except TypeError:
            # 舊版沒有 with_score
            docs = vector_store.similarity_search(question, k=k, expr=expr)
            return [(d, 0.0) for d in docs]
        except Exception as e:
            msg = str(e).lower()
            if "field" in msg and "not exist" in msg:
                # 2) fallback: metadata["source"]
                expr2 = build_expr(ids, use_json=True)
                try:
                    return vector_store.similarity_search_with_score(question, k=k, expr=expr2)
                except TypeError:
                    docs = vector_store.similarity_search(question, k=k, expr=expr2)
                    return [(d, 0.0) for d in docs]
            # 其他錯誤就拋出
            raise

    try:
        docs = []

        # --- 多檔：每檔各取一些、再合併去重排序 ---
        if req.file_ids and len(req.file_ids) > 1:
            k_per = max(2, math.ceil(req.k / len(req.file_ids)))
            pool = {}
            for fid in req.file_ids:
                results = await search_with_fallback(req.question, k_per, [fid])
                for d, s in results:
                    key = (
                        d.metadata.get("source"),
                        d.metadata.get("page"),
                        d.metadata.get("chunk_in_page"),
                    )
                    if key not in pool or s < pool[key][1]:
                        pool[key] = (d, s)
            docs = [d for d, _ in sorted(pool.values(), key=lambda x: x[1])[: req.k]]

        # --- 單檔：直接在該檔案範圍檢索 ---
        elif req.file_ids:
            results = await search_with_fallback(req.question, req.k, req.file_ids)
            docs = [d for d, _ in results]

        # --- 未指定檔案：維持原本全庫檢索 ---
        else:
            try:
                results = vector_store.similarity_search_with_score(req.question, k=req.k)
                docs = [d for d, _ in results]
            except TypeError:
                docs = vector_store.similarity_search(req.question, k=req.k)

        # 組合文字 context
        context = "\n".join(d.page_content for d in docs)

        img_paths, seen = [], set()
        for d in docs:
            src = d.metadata.get("source")
            page = d.metadata.get("page")
            key = (src, page)
            if key in seen:
                continue
            seen.add(key)
            img_field = d.metadata.get("images", "")
            if img_field and img_field.strip():
                img_paths.extend(p for p in img_field.split(";") if p)

        return SearchResponse(
            context=context,
            image_paths=img_paths,
            sources=[d.metadata for d in docs],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/delete_source")
async def delete_source(req: DeleteSourceRequest):
    try:

        if not utility.has_collection(COLLECTION):
            return {"status": "ok", "deleted": 0}

        coll = Collection(COLLECTION)
        coll.load()  # 確保可操作
        expr = f'source == "{req.source_filename}"'
        mr = coll.delete(expr)  
        return {"status": "ok", "expr": expr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus delete failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)