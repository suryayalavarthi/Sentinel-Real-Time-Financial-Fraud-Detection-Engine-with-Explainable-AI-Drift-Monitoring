# System Blueprint: suryayalavarthi/Sentinel-Real-Time-Financial-Fraud-Detection-Engine

> 🛡️ High-frequency Fraud Detection Engine with real-time inference (6-8ms) and automated MLOps drift monitoring. $730K+ estimated annual ROI.
>
> Auto-generated on 2026-02-13 by Repo-to-Blueprint Architect

## Project Purpose
Real-time financial fraud detection engine processing transaction data with 6-8ms inference latency. Uses XGBoost models served via NVIDIA Triton, with SHAP-based explainability for high-risk predictions and streaming drift monitoring for production MLOps.

## Technical Stack
- **Language**: Python 3.11
- **Framework**: FastAPI (serving), XGBoost (ML), NVIDIA Triton Inference Server (model serving)
- **Key Dependencies**:
  - ML: `xgboost>=1.5.0`, `scikit-learn>=1.0.0`, `shap>=0.40.0`
  - Serving: `fastapi>=0.110.0`, `uvicorn>=0.27.0`, `tritonclient[grpc,http]>=2.34.0`
  - Quantization: `hummingbird-ml==0.4.12`, `onnx>=1.14.0`, `onnxruntime>=1.15.0`
  - Monitoring: `prometheus-client>=0.19.0`, `scipy>=1.7.0` (drift detection)
- **Infrastructure**: Docker (Dockerfile, Dockerfile.gateway, Dockerfile.triton), Docker Compose, GitHub Actions CI/CD, Prometheus, Grafana

## Architecture Blueprint

```mermaid
flowchart TD
    subgraph Client["Client Layer"]
        CLI["API Client"]
    end

    subgraph Gateway["Gateway Service (FastAPI)"]
        API["main.py / src/api.py"]
        XAI["src/xai.py<br/>(SHAP Explainer)"]
        TCLIENT["src/triton_client.py"]
    end

    subgraph Inference["Inference Layer"]
        TRITON["Triton Server<br/>(ONNX Model)"]
        REPO[("model_repository/<br/>sentinel_model")]
    end

    subgraph MLOps["MLOps Pipeline"]
        TRAIN["src/model_training.py"]
        EVAL["src/model_evaluation.py"]
        QUANT["quantize/quantize.py<br/>(XGBoost→ONNX)"]
        FEAT["src/feature_engineering.py"]
        INGEST["src/data_ingestion.py"]
    end

    subgraph Monitoring["Monitoring Stack"]
        DRIFT["monitoring/drift.py<br/>(Streaming Drift)"]
        PROM["Prometheus"]
        GRAF["Grafana"]
        SCRIPT["scripts/monitoring.py"]
    end

    subgraph Storage["Artifact Storage"]
        MODELS[("models/<br/>sentinel_fraud_model.pkl<br/>feature_names.json")]
        DATA[("ieee-fraud-detection/<br/>processed/")]
        LOGS[("logs/")]
    end

    CLI -->|"POST /predict"| API
    API --> TCLIENT
    API --> XAI
    TCLIENT -->|"gRPC"| TRITON
    TRITON --> REPO

    INGEST --> DATA
    DATA --> FEAT
    FEAT --> TRAIN
    TRAIN --> MODELS
    TRAIN --> EVAL
    MODELS --> QUANT
    QUANT --> REPO

    API -->|"metrics"| PROM
    SCRIPT --> DRIFT
    DRIFT --> DATA
    DRIFT --> LOGS
    PROM --> GRAF

    style API fill:#1f6feb,stroke:#58a6ff,color:#fff
    style TRITON fill:#238636,stroke:#3fb950,color:#fff
    style REPO fill:#da3633,stroke:#f85149,color:#fff
    style MODELS fill:#da3633,stroke:#f85149,color:#fff
    style DATA fill:#da3633,stroke:#f85149,color:#fff
    style PROM fill:#8b949e,stroke:#c9d1d9,color:#fff
    style GRAF fill:#8b949e,stroke:#c9d1d9,color:#fff

```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Gateway as Gateway (main.py)
    participant TritonClient as TritonClient
    participant Triton as Triton Server
    participant XAI as SHAP Explainer
    participant Prometheus

    Client->>Gateway: POST /predict {features}
    activate Gateway
    Gateway->>Gateway: Validate feature_names.json
    Gateway->>TritonClient: infer(feature_vector)
    activate TritonClient
    TritonClient->>Triton: gRPC ModelInfer (ONNX)
    activate Triton
    Triton-->>TritonClient: fraud_probability
    deactivate Triton
    TritonClient-->>Gateway: probability
    deactivate TritonClient

    alt High Risk (prob >= 0.5 or confidence=High)
        Gateway->>XAI: explain(features, probability)
        activate XAI
        XAI-->>Gateway: top_features (SHAP values)
        deactivate XAI
    end

    Gateway->>Prometheus: Increment PREDICT_REQUESTS
    Gateway->>Prometheus: Observe REQUEST_LATENCY
    Gateway-->>Client: {fraud_probability, is_fraud, rationale}
    deactivate Gateway

```

## Evidence-Based Risks

1. **Hardcoded Threshold (0.5)**: `main.py:23` defines `THRESHOLD = 0.5` globally; no dynamic calibration mechanism found in codebase, limiting adaptability to evolving fraud patterns.

2. **Silent Fallback on Triton Failure**: `main.py:88-92` catches all exceptions during `TritonClient` initialization and sets `triton_client = None`, causing `/predict` endpoint to raise HTTP 500 (`main.py:135`) without retry logic or circuit breaker.

3. **Unbounded Drift Buffer**: `monitoring/drift.py` (referenced in `docker-compose.yml:67` and `main.py:96`) uses `buffer_size=10000` with no eviction policy visible in file tree; prolonged operation risks memory exhaustion.

4. **Missing Model Versioning**: `model_repository/sentinel_model/config.pbtxt` and `models/model_metadata.json` exist but no version control mechanism in `src/model_training.py` or deployment scripts; rollback requires manual file replacement.

5. **CI Model Generation Bypasses Validation**: `scripts/generate_ci_model.py` (used in `.github/workflows/smoke-test.yml:72`) creates minimal ONNX models for testing but smoke test doesn't validate prediction accuracy against known fraud cases, risking deployment of undertrained models.

---

## Repository Stats
| Metric | Value |
|--------|-------|
| Total Files | 57 |
| Total Directories | 13 |
| Generated | 2026-02-13 |
| Source | [suryayalavarthi/Sentinel-Real-Time-Financial-Fraud-Detection-Engine](https://github.com/suryayalavarthi/Sentinel-Real-Time-Financial-Fraud-Detection-Engine) |

---

*Generated by Repo-to-Blueprint Architect via n8n*
