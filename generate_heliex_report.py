#!/usr/bin/env python3
"""
Generate Heliex system report (Markdown with base64-embedded diagrams),
optionally export to PDF.

Usage examples:
  python generate_heliex_report.py
  python generate_heliex_report.py --arch-png heliex_architecture.png --lora-png heliex_lora_workflow.png
  python generate_heliex_report.py --arch-b64 architecture.b64 --lora-b64 lora.b64
  python generate_heliex_report.py --out-md heliex_report.md --out-pdf heliex_report.pdf
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
from textwrap import dedent

def guess_paths():
    """
    Return best-guess paths for diagrams:
    - Prefer PNGs named heliex_architecture.png and heliex_lora_workflow.png
    - Else fall back to architecture.b64 and lora.b64
    """
    candidates = {
        "arch_png": "heliex_architecture.png",
        "lora_png": "heliex_lora_workflow.png",
        "arch_b64": "architecture.b64",
        "lora_b64": "lora.b64",
    }
    found = {}
    for k, v in candidates.items():
        found[k] = v if os.path.exists(v) else None
    return found

def read_b64_from_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_b64_from_text(path: str) -> str:
    # Reads entire file (even if multi-line). Strips trailing whitespace.
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    # If the user pasted headers/footers like "---END OF lora---", drop them.
    # Keep only base64-looking chars.
    import re
    data = "".join(re.findall(r"[A-Za-z0-9+/=\n\r]+", data))
    # Remove newlines to make the data: URL compact.
    return data.replace("\n", "").replace("\r", "")

def resolve_b64(arch_png, lora_png, arch_b64, lora_b64):
    g = guess_paths()

    # Architecture
    if arch_b64:
        arch_data = read_b64_from_text(arch_b64)
    elif arch_png:
        arch_data = read_b64_from_png(arch_png)
    elif g["arch_png"]:
        arch_data = read_b64_from_png(g["arch_png"])
    elif g["arch_b64"]:
        arch_data = read_b64_from_text(g["arch_b64"])
    else:
        sys.exit("❌ Could not find architecture diagram. Provide --arch-png or --arch-b64.")

    # LoRA workflow
    if lora_b64:
        lora_data = read_b64_from_text(lora_b64)
    elif lora_png:
        lora_data = read_b64_from_png(lora_png)
    elif g["lora_png"]:
        lora_data = read_b64_from_png(g["lora_png"])
    elif g["lora_b64"]:
        lora_data = read_b64_from_text(g["lora_b64"])
    else:
        sys.exit("❌ Could not find LoRA workflow diagram. Provide --lora-png or --lora-b64.")

    return arch_data, lora_data

def build_markdown(arch_b64: str, lora_b64: str) -> str:
    # Full, single-file Markdown with embedded PNGs (base64).
    # Adjust content as needed; this matches the latest agreed report (executive summary,
    # architecture, developer deep dives, LLM comparison, costs).
    md = f"""\
# 🧠 Heliex: LLM-Powered Empathic Companion for Mental Health Support  
### *Technical Architecture, LoRA Personalization, and Cost Evaluation for a Medium-Sized Startup (2025)*

---

## 🧩 Executive Summary
**Heliex** is an AI-powered mental-health platform designed to serve as an *empathic companion* with a psychotherapy background.  
It bridges the gap between human therapists and AI systems by emulating professional communication styles through LLM-driven personas.  
Each registered psychologist provides a brief description of their therapeutic approach and can optionally fine-tune an LLM persona via a lightweight **LoRA (Low-Rank Adaptation)** adapter trained on anonymized dialogue data.  

Heliex leverages a **modular, cloud-native architecture** combining safety validation, real-time inference, and dynamic persona loading.  
This approach enables both scalability and deep personalization while adhering to clinical safety and data privacy requirements.

---

## 1. System Architecture

![Heliex System Architecture](data:image/png;base64,{arch_b64})

### **Key Components**
| Layer | Description |
|-------|--------------|
| **Frontend (Client App)** | Web/mobile chat interface for users and psychologists. |
| **API Gateway / Proxy** | Manages authentication, routing, and rate limiting. |
| **Persona Orchestration Layer** | Selects the appropriate persona prompt or loads a LoRA adapter; applies guardrails. |
| **Persona Knowledge Base (KB)** | Stores psychologist profiles, summaries, example dialogues, and LoRA adapter checkpoints. |
| **Core LLM Backend** | Performs inference using the base model plus optional LoRA adapter injection. |
| **Post-Processor** | Applies final safety filters and compliance logging before delivering responses. |

This layered structure ensures flexibility, maintainability, and secure separation between user-facing services and sensitive AI components.

---

## 2. LoRA Adapter Personalization

**Low-Rank Adaptation (LoRA)** is a fine-tuning method that adds small trainable matrices inside transformer layers, leaving base weights frozen.  
Each psychologist’s tone and interaction style can be learned efficiently with LoRA adapters that typically require **under 200 MB of storage**.

### **Heliex LoRA Workflow**

![Heliex LoRA Workflow](data:image/png;base64,{lora_b64})

**Advantages**
- Extremely lightweight and cost-efficient.  
- Enables real-time persona switching.  
- Preserves base model quality and safety compliance.  
- Scalable to hundreds of unique psychologist profiles.

---

## 3. Developer Implementation Deep-Dives

### 3.1 Data Preprocessing
**Objectives:** anonymize, standardize, and structure dialogue data for training.

**Pipeline:**
1. **Anonymization** — NER-based redaction for PII (names, locations, contact info), plus optional human spot checks.  
2. **Segmentation** — Split conversations into `(user, therapist)` pairs; filter very short turns.  
3. **Normalization** — Unicode normalize, normalize whitespace, strip system artifacts.  
4. **Quality Filters** — Deduplicate by hash, remove toxic/off-topic segments unless explicitly permitted for safety tests.  
5. **Tokenization** — Use the base model’s tokenizer; enforce max sequence length; truncate or sliding-window long sessions.  
6. **Formatting** — Store as JSONL or Parquet with fields: `persona_id`, `input`, `output`, `tags`, `timestamp`.  

**Infra Notes (cloud):**
- Stateless CPU workers on AWS Fargate / GCP Cloud Run.  
- Throughput: ~3 min / 100 convos CPU-side.  
- Store clean datasets in object storage (S3/GS) with versioning.

---

### 3.2 LoRA Training
**Frameworks:** Hugging Face `peft`, PyTorch Lightning LoRA.  
**Base Models:** OpenAI fine-tunable endpoint (if available for your plan) or OSS (Mistral 7B, Llama 3 8B).  

**Typical Hyperparameters:**
- `r = 8` (rank), `lora_alpha = 16`, `lora_dropout = 0.1`  
- `lr = 2e-4`, batch size 16, epochs 3–5  
- Optimizer: AdamW (weight decay 0.01)  
- Precision: bfloat16/float16

**Artifacts:** Only adapter weights (e.g., `.safetensors`) — ~50–200 MB.

**Cost/Time (AWS A100 spot):**
- ≈ **$18–$25 per run** (≤ 1 hour), depending on dataset size.

**Operational Flow (on-demand per psychologist):**
1. Psychologist triggers training from dashboard.  
2. Preprocessed dataset is fetched (`persona_id`).  
3. Launch short-lived GPU job (EKS/SageMaker/Batch).  
4. Save adapter + metrics, update Persona KB index.  

---

### 3.3 Validation & Safety QA
**Automated:**
- **Toxicity & Self-harm Classifiers:** reject unsafe generations.  
- **GuardrailsAI LlmRagEvaluator:** detect hallucinations and factual drift.  
- **Jailbreak Detection:** pattern-based + LLM meta-detectors.

**Manual:**
- **Clinician Review:** sample 10–20 prompts; rate empathy, clarity, appropriateness on Likert scales.  
- **Decision:** approve, quarantine, or request retraining.

**Release Gate:** Adapters not passing automated + manual checks are **not** promoted to production.

---

### 3.4 Client Inference
**Runtime Steps:**
1. User selects a psychologist persona.  
2. **Orchestrator**:  
   - Retrieves persona prompt + (optional) RAG snippets + **LoRA adapter**.  
   - Assembles a constrained system prompt (ethics & scope).  
3. **Core LLM** runs inference with the adapter injected.  
4. **Post-Processor** applies final safety filters; logs metadata.

**Latency Targets:** 2–3 s per message with caching and streaming; LoRA overhead is minimal when adapters are preloaded.

---

### 3.5 Monitoring & Retraining
**What to Track:**
- Safety flags, jailbreak attempts, refusal rates.  
- Session satisfaction (thumbs up/down), NPS proxy.  
- Persona usage distribution; abnormally high/low engagement.

**Drift & Retraining Triggers:**
- **Automatic:** drift > 15% vs baseline on semantic clusters or sentiment.  
- **Manual:** psychologist uploads new examples or requests updates.

**Cadence:** on-demand per psychologist; common cadence monthly or bi-monthly for active personas.

---

## 4. LLM Selection Strategies

### 4.1 OpenAI ChatGPT API
- **Pros:** zero infra, enterprise-grade compliance, predictable latency, fast iteration.  
- **Cons:** limited weight-level control; LoRA only if/when supported by provider; data residency considerations.

### 4.2 Open-Source LLMs (Mistral, Llama 3, Falcon, etc.)
- **Pros:** full control of weights/adapters; flexible privacy (VPC); offline/on-prem possible; tunable latency-cost tradeoffs.  
- **Cons:** GPU infra + MLOps needed (K8s/EKS, autoscaling, logging, model registry); higher ongoing ops overhead.

---

### **Comparison Table**

| Feature | OpenAI ChatGPT | Open-Source LLM |
|---|---|---|
| **Hosting** | Managed API | Self-hosted (AWS/GCP, EKS/SageMaker/Vertex) |
| **Infra Cost (monthly)** | $0 setup | **$3,000–$5,000** (1× A100 + support) |
| **Per-Message Cost** | API usage-based (~$1,000–1,500 / 100k msgs) | GPU time + storage; can be cheaper at high volume |
| **Latency** | ~2 s | ~3–5 s (typical), tunable |
| **Customization** | Prompts, structured tools | Full LoRA fine-tuning + control |
| **Security/Privacy** | External processing, enterprise programs available | Data stays in VPC / on-prem |
| **Compliance** | SOC2/HIPAA (provider program) | Your responsibility (audit, logging, access controls) |
| **Maintenance** | Low | High (DevOps + ML Ops) |

---

## 5. Development Effort & Cost Estimation (USD)

**Context:** Medium-sized startup, cloud-native, on-demand LoRA per psychologist, 6-month MVP→Prod.

### 5.1 Dev Effort (One-time, team of 5)
| Phase | Scope | Effort | Cost |
|---|---|---:|---:|
| Architecture & CI/CD | Infra design, pipelines, IaC | 6 wks | $25,000 |
| Frontend & Gateway | Web/mobile chat, auth, routing | 8 wks | $35,000 |
| Persona Orchestrator & KB | DB schemas, prompt mgmt, adapter registry | 8 wks | $30,000 |
| LoRA Pipeline | Training service + validation + storage | 10 wks | $45,000 |
| QA & Security | Safety tests, compliance review | 6 wks | $20,000 |
| Deploy & Monitoring | Observability, rollouts | 6 wks | $25,000 |
| **Total** | ~6 months | — | **$180,000–$200,000** |

### 5.2 Monthly Ops Costs
| Item | ChatGPT API | Open-Source LLM |
|---|---:|---:|
| Inference (100k msgs) | $1,000–1,500 | — |
| LoRA Training (on-demand) | $200–500 | $800–1,200 |
| Storage (S3/GS) | $100 | $300 |
| Monitoring & Ops | $200 | $1,000 |
| Infra (GPU/K8s) | $0 | $3,000–5,000 |
| **Total / month** | **$1,500–2,300** | **$5,000–8,000** |

> **Takeaway:** Start on ChatGPT API for speed & cost efficiency; migrate to a hybrid/open-source stack as volume or privacy demands grow.

---

## 6. Strategic Conclusion

Heliex delivers **scalable, ethical, personalized** mental-health conversations by blending a shared LLM backbone with **persona prompts** and optional **LoRA adapters**.  
The architecture supports rapid MVP on managed APIs and evolves naturally to a private-cloud/open-source stack when needed — balancing cost, control, and compliance across growth stages.

**End of Report**  
*(Heliex 2025 — Confidential Technical Architecture Document)*
"""
    return dedent(md)

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def try_make_pdf(md_path: str, pdf_path: str) -> bool:
    """
    Try to create a PDF using pandoc or weasyprint (whichever is installed).
    Returns True if PDF created, False otherwise.
    """
    # Option 1: pandoc + xelatex if available
    if shutil.which("pandoc"):
        cmd = ["pandoc", md_path, "-o", pdf_path, "--pdf-engine=xelatex", "-V", "geometry:margin=1in"]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(f"pandoc failed: {e}")

    # Option 2: weasyprint (HTML-based)
    if shutil.which("weasyprint"):
        # Convert MD to HTML via Python (very minimal) then to PDF
        try:
            import markdown  # pip install markdown
            html = markdown.markdown(open(md_path, "r", encoding="utf-8").read(), extensions=["tables"])
            tmp_html = md_path.replace(".md", ".tmp.html")
            write_text(tmp_html, "<meta charset='utf-8'>\n" + html)
            subprocess.run(["weasyprint", tmp_html, pdf_path], check=True)
            os.remove(tmp_html)
            return True
        except Exception as e:
            print(f"weasyprint failed: {e}")

    return False

def main():
    parser = argparse.ArgumentParser(description="Generate Heliex report (Markdown with base64 diagrams); optional PDF export.")
    parser.add_argument("--arch-png", type=str, default=None, help="Path to heliex_architecture.png")
    parser.add_argument("--lora-png", type=str, default=None, help="Path to heliex_lora_workflow.png")
    parser.add_argument("--arch-b64", type=str, default=None, help="Path to architecture.b64 (base64 text)")
    parser.add_argument("--lora-b64", type=str, default=None, help="Path to lora.b64 (base64 text)")
    parser.add_argument("--out-md", type=str, default="heliex_report.md", help="Output Markdown filename")
    parser.add_argument("--out-pdf", type=str, default=None, help="Optional PDF output filename")
    args = parser.parse_args()

    arch_b64, lora_b64 = resolve_b64(args.arch_png, args.lora_png, args.arch_b64, args.lora_b64)
    md = build_markdown(arch_b64, lora_b64)
    write_text(args.out_md, md)
    print(f"✅ Wrote Markdown: {args.out_md}")

    if args.out_pdf:
        ok = try_make_pdf(args.out_md, args.out_pdf)
        if ok:
            print(f"📄 Wrote PDF: {args.out_pdf}")
        else:
            print("⚠️ PDF generation skipped/failed. Install pandoc (and xelatex) or weasyprint to enable PDF export.")

if __name__ == "__main__":
    main()
