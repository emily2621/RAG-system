# One-click deploy, multimodal RAG system with a visual file selector


## 架構總覽

1. 使用者在前端 `index.html` 上傳 PDF、勾選檔案並提問
2. 前端透過 **JSON-RPC** 呼叫 `MCP Server`（:8003）
3. `MCP Server` 會：

   * 交給 `Host App`（:8000）做 `/ingest`（內部再去叫 **PDF Tool** 轉檔與 **RAG Tool** 入庫）與 `/query` 推論。&#x20;
4. **PDF Tool**（:8001）把 PDF 轉成 Markdown、擷取**原圖**與**縮圖**並落地到 `pdfs/images` 與 `pdfs/thumbs`。
5. **RAG Tool**（:8002）把 Markdown 切塊後寫入 **Milvus**（collection: `doc_chunks_mcp`），`/search` 可依 \*\*file\_ids（= 檔名）\*\*過濾來源，只檢索被勾選的檔案。
6. `Host App` 以 **Qwen/Qwen2.5-VL-7B-Instruct** 做多模態生成，將檢索回來的文字 **+ 相關頁面縮圖**一併餵給模型，最後回傳答案與來源。

> 靜態縮圖 `/thumbs/*` 與原圖 `/images/*` 由 `MCP Server` 對外提供；首頁 UI 也是由它服務。&#x20;

---

## 服務與連接埠

* **Host App**（FastAPI）: `http://localhost:8000`（/ingest, /query）
* **PDF Tool**: `http://localhost:8001`（/process\_pdf）&#x20;
* **RAG Tool**: `http://localhost:8002`（/ingest, /search, /delete\_source）
* **MCP Server + 前端頁面**: `http://localhost:8003`（`/`、`/mcp/v1/rpc`、`/thumbs/*`、`/images/*`）&#x20;
* **Milvus**（建議以 Docker 啟動）：gRPC `:19530`、health check `http://localhost:9091/healthz`（Makefile 已內建檢查）

各服務的對內 URL 也可透過環境變數調整（`PDF_TOOL_URL`, `RAG_TOOL_URL`, `HOST_URL`）。&#x20;

---

## 使用 Docker 快速部署

### 前置需求

* 建議 **NVIDIA GPU + 驅動 + CUDA 容器環境**（模型與嵌入均支援 GPU）&#x20;
* 已安裝 **Docker** 與 **docker compose**
* 首次啟動需能連網以拉取 HuggingFace 模型（Qwen2.5-VL 與 `intfloat/multilingual-e5-large`）&#x20;

### 一鍵啟動（專案根目錄）

```bash
# 第一次建 base image + 啟動所有服務（或直接 make up）
make up-fast
# health check（Milvus / Host App / PDF Tool / RAG Tool / MCP Web）
make health
```

### 常用指令

```bash
make up            # 啟動/重建
make down          # 關閉
make logs          # 查看日誌
make ps            # 看容器狀態
make restart       # 重新啟動（down -> up-fast）
make build-base    # 只重建 Py3.12 base（三個服務共用）
make build-host    # 只重建 Host App（改 requirements 或 Dockerfile 時）
```

### 打開前端

* 造訪：`http://localhost:8003/`（首頁有「上傳 PDF、勾選檔案、提問」UI）

---

## UI操作流程

1. **上傳 PDF**：按「Upload」後，系統會：

   * 把 PDF 丟給 `Host App` 的 `/ingest`
   * `Host App` 轉叫 `PDF Tool` `/process_pdf` 產生 Markdown、表格與圖像；接著呼叫 `RAG Tool` `/ingest` 入庫。 
    
    <img width="1918" height="867" alt="image" src="https://github.com/user-attachments/assets/0e9dc262-c29d-4a3d-badf-1aa5685a4c66" />

2. **勾選檔案**：每個成功上傳的檔名會出現在清單，**打勾代表稍後只檢索這些檔案**（file\_ids）。勾選checkbox可選擇**問指定檔案問題**或**刪除選取的檔案**<img width="1919" height="680" alt="image" src="https://github.com/user-attachments/assets/58b35695-dd9f-4e4c-a0f7-e598d4008ac2" />

3. **輸入問題並送出**：前端呼叫 `MCP Server` 的 `predict` 方法；伺服器端會只在被勾選的檔案範圍內做相似度檢索，並將**同頁的縮圖**與文字上下文一起送給多模態模型生成答案，並且會回傳source metadata以顯示資料來源。
    
    <img width="1919" height="876" alt="image" src="https://github.com/user-attachments/assets/6ee7e22d-308d-4b5e-a609-9ab85329e8a5" />

4. **刪除檔案（可選）**：支援刪 PDF 原檔 / 轉檔（md、images、thumbs）以及 **Milvus 中該來源的所有 chunks**。勾選checkbox即可選擇欲刪除檔案。<img width="1919" height="680" alt="image" src="https://github.com/user-attachments/assets/43ffcadf-bc6d-465c-960e-9ee1b1465edf" />


---

## 為什麼問答能對到「正確圖片」？

* **入庫時的 metadata**：每個文字 chunk 都會帶上
  `source`（檔名）、`page`（頁碼）、`chunk_in_page`、以及該頁 **images 的檔案清單**（以 `;` 連結）。若該頁**沒有文字但有圖片**，會補上一個 **sentinel chunk**，確保純圖片頁也能被檢索到。
* **檢索回傳**：`/search` 會把相同 `(source, page)` 的圖片去重、回傳一份 `image_paths`。
* **推論前處理**：`Host App` 會組 prompt：把**文字 context** + **相對應頁面的縮圖 URL** 一起餵給 **Qwen2.5-VL**。
* **前端只搜被勾選的檔案**：UI 會把 checkbox 對應的 `file_ids` 傳進 `predict`，後端用 `source == "<檔名>"` 的條件在 Milvus 篩選（舊版欄位不存在時，自動回退到 `metadata["source"]` 表示法）。&#x20;

---

## 檔案與資料夾

* `pdfs/`：上傳與轉檔後的 Markdown
* `pdfs/images/`：擷取的**原始圖片**（檔名格式：`<pdf>_page<i>_img<j>.<ext>`）
* `pdfs/thumbs/`：對應的**縮圖**（`..._thumb.jpg`）


---

## 串接指南（兩種方式）

### A. 走前端 JSON-RPC（建議）

`POST /mcp/v1/rpc`，`method` 可用：

* `uploadPdf`、`processPdf`、`ingest`、`search`、`query`、`predict`、`deleteSource`
* `listTools`、`listResources`、`listPrompts`、`updatePrompt`（修改提示詞模板）

**範例：predict（只搜兩個檔案）**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "predict",
  "params": {
    "file_ids": ["AAA.pdf", "BBB.pdf"],
    "question": "請解釋圖1的流程，並指出步驟順序",
    "k": 15
  }
}
```

### B. 直接打微服務（REST）

* **PDF Tool**

  * `POST /process_pdf`：`{"pdf_path": "/abs/path/to/file.pdf"}` → 回 Markdown 路徑、每頁圖片/表格清單
* **RAG Tool**

  * `POST /ingest`：帶上 `md_path`、每頁 `images/tables`、`source_filename` 寫入 Milvus
  * `POST /search`：`{"question":"...","k":15,"file_ids":["foo.pdf"]}`（可選 `file_ids` 做來源過濾）
  * `POST /delete_source`：刪除某檔名的所有向量（`source == "<檔名>"`）
* **Host App**

  * `POST /ingest`：表單檔案上傳；內部會連 PDF Tool → RAG Tool 完成整個 ingest pipeline
  * `POST /query`：`{"question":"...","file_ids":[...]}`
    （會先呼叫 RAG Tool `/search` 拿 context 與同頁縮圖，再多模態生成）

---

## 重要細節與可調項

* **提示詞模板（Prompts）**：`Host App` 啟動時會向 `MCP Server` 取得模板，預設回覆語言為繁中、需指引用到的圖表並描述。可用 `updatePrompt` 動態更新。&#x20;
* **模型**：`Qwen/Qwen2.5-VL-7B-Instruct`，以 `AutoModelForVision2Seq` + `AutoProcessor` 載入（FP16 / `device_map="auto"`）；推論 `max_new_tokens=1024`。
* **向量化**：`intfloat/multilingual-e5-large`，`normalize_embeddings=True`，預設走 CUDA。
* **Milvus Collection**：`doc_chunks_mcp`（可在程式碼中修改）。
* **分塊策略**：`chunk_size=800, overlap=200`，以頁為單位切段再分塊。
* **PDF 解析路徑**：先試 `markitdown` → 無則用 PyMuPDF 取文字 → 太短再 OCR（`chi_tra+eng`）。同時擷取原圖並產生縮圖。

---

## 除錯與health check

* **一次看全部 log **：`make logs`
* **health check 指令**：`make health`（Milvus / Host App / PDF Tool / RAG Tool / MCP Web）
* **典型錯誤排查**

  * **Milvus 未就緒**：`http://localhost:9091/healthz` 應為 OK；否則等服務起來再重試。
  * **模型載入卡住/顯存不足**：檢查 GPU；必要時改為 CPU（速度較慢）。
  * **PDF 轉檔品質差**：安裝/檢查 `markitdown`；或確認容器內 Tesseract 與 `chi_tra`/`eng` 語言包可用（OCR fallback）。
  * **找不到圖片**：確認入庫 metadata 是否含該頁的 `images`，與 `/search` 回傳的 `image_paths` 是否為同頁縮圖。
  * **只想檢索特定檔案**：確保前端勾選有帶出 `file_ids`；RAG Tool `/search` 會依 `source == "<檔名>"` 過濾（舊版欄位名會自動 fallback）。&#x20;

---

## 開發者工作流建議

* **只改 Python 程式碼**：多半直接 hot-reload；若容器未設定 reload，`make restart` 最快。
* **改了 `requirements` 或 Dockerfile**：`make build-host` 後 `make up-fast`。
* **重建 Base 層（較少用）**：`make build-base`（三個 Py3.12\_base 服務共用）。
* **環境變數**：需要時於 Compose 或容器環境覆寫 `PDF_TOOL_URL`、`RAG_TOOL_URL`、`HOST_URL`。&#x20;

---

## API 快速測試（cURL）

**查詢（只搜某檔）**

```bash
curl -s http://localhost:8002/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is Figure 1 about?","k":15,"file_ids":["your.pdf"]}'
```

**多模態推論（直接打 Host App）**

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"請總結圖2重點","file_ids":["your.pdf"]}'
```

（正式建議還是透過 `MCP Server` 的 `predict`，它會幫你串完搜尋與推論。）&#x20;

---

## 安全清理

**刪某份上傳檔的所有痕跡（檔案 + Milvus）**

* 前端 UI：勾選後按「Delete Selected」
* 後端：`POST /mcp/v1/rpc` method=`deleteSource` → 同步刪掉 `pdfs/*.pdf/.md`、`images/`、`thumbs/` 與 Milvus 中 `source == "<檔名>"` 的向量。&#x20;

---

### 小結

* **可視化選檔**避免跨檔汙染檢索
* **圖片鏈接回頁面**，多模態模型能「看圖說話」
* **單機一鍵啟動**，`make health` 快速驗證
* **JSON-RPC + REST** 兩種串接方式，易於整合既有系統
