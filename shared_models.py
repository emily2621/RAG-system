# # shared_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Union

# --- Upload Response ---
class UploadResponse(BaseModel):
    file_id: str  # Uploaded PDF identifier

# --- PDF Processing Models ---
class ProcessRequest(BaseModel):
    pdf_path: str

class ProcessResponse(BaseModel):
    md_path: str
    images: Dict[int, List[str]]       # page number -> list of image paths
    tables: Dict[int, List[str]]       # page number -> list of markdown tables

# --- RAG Tool Models ---
class IngestRequest(BaseModel):
    md_path: str
    images: Dict[int, List[str]]
    tables: Dict[int, List[str]]
    source_filename: str

class SearchRequest(BaseModel):
    question: str
    k: int = 15
    file_ids: Optional[List[str]] = None

class SearchResponse(BaseModel):
    context: str
    image_paths: List[str]
    sources: List[Dict[str, Any]]       # metadata dicts for each chunk

# --- MCP Unified Models ---
class Reference(BaseModel):
    page: int
    type: Literal['text', 'image', 'table']
    name: Optional[str] = None
    url: Optional[str] = None
    markdown: Optional[str] = None

class Output(BaseModel):
    text: str
    references: List[Reference]

class MCPRequest(BaseModel):
    file_ids: List[str]
    question: str
    params: Dict[str, Any] = Field(default_factory=dict)

class MCPResponse(BaseModel):
    output: Output

# --- MCP JSON-RPC Discovery Models ---
class ToolDescriptor(BaseModel):
    name: str
    description: Optional[str] = None
    params_schema: Dict[str, Any]

class ResourceDescriptor(BaseModel):
    name: str
    type: Literal['static', 'dynamic']
    description: Optional[str] = None
    url: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

class PromptTemplate(BaseModel):
    name: str
    template: str

class ListToolsResponse(BaseModel):
    tools: List[ToolDescriptor]

class ListResourcesResponse(BaseModel):
    resources: List[ResourceDescriptor]

class ListPromptsResponse(BaseModel):
    prompts: List[PromptTemplate]

class UpdatePromptRequest(BaseModel):
    name: str
    template: str

class UpdatePromptResponse(BaseModel):
    success: bool
    prompt: PromptTemplate
