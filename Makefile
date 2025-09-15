.PHONY: up up-fast build-base build-host down logs ps restart health

# 讓 Docker BuildKit & compose-bake 預設啟用（加速建置）
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

COMPOSE_FILE := deploy/docker-compose.hostnet.yml

# 先建共用 base（只需偶爾重建；三個 Py3.12_base service 會共用這層套件）
build-base:
	docker compose -f $(COMPOSE_FILE) build py312_base

# 只重建 host_app（改 requirements 或 host_app Dockerfile 時才需要）
build-host:
	docker compose -f $(COMPOSE_FILE) build host_app

# 一鍵啟動（自動建 base 再 up）
up-fast: build-base
	docker compose -f $(COMPOSE_FILE) up -d --build

# 傳統 up（如果已建過 base，也可以用這個）
up:
	docker compose -f $(COMPOSE_FILE) up -d --build

down:
	docker compose -f $(COMPOSE_FILE) down || true

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

ps:
	docker compose -f $(COMPOSE_FILE) ps

restart: down up-fast

# health check
health:
	@curl -sSf http://localhost:9091/healthz >/dev/null && echo "Milvus OK" || (echo "Milvus NOT ready"; exit 1)
	@curl -sSf http://localhost:8000/docs >/dev/null && echo "Host App OK" || (echo "Host App NOT ready"; exit 1)
	@curl -sSf http://localhost:8001/docs >/dev/null && echo "PDF Tool OK" || (echo "PDF Tool NOT ready"; exit 1)
	@curl -sSf http://localhost:8002/docs >/dev/null && echo "RAG Tool OK" || (echo "RAG Tool NOT ready"; exit 1)
	@curl -sSf http://localhost:8003 >/dev/null && echo "MCP Web OK" || (echo "MCP Web NOT ready"; exit 1)
