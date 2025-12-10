import base64
from graphviz import Digraph

# === 1. Generate Heliex Architecture Diagram ===
arch = Digraph("Heliex_System_Architecture", format="png")
arch.attr(rankdir="TB", size="8")

arch.node("client", "Client App\n(Web/Mobile Chat)", shape="box", style="filled,rounded", fillcolor="#E6F2FF")
arch.node("gateway", "API Gateway / Proxy\n(Routes Requests, Auth, Rate Limits)", shape="box", style="filled,rounded", fillcolor="#F2F2F2")
arch.node("orchestrator", "Persona & Safety Orchestration Layer\n- Persona Selector (Prompt, RAG, LoRA)\n- Safety Guardrails\n- Dialogue Manager", shape="box", style="filled,rounded", fillcolor="#FFF2CC")
arch.node("kb", "Persona Knowledge Base\n- Summary Prompts\n- Example Dialogues (RAG)\n- LoRA Adapter", shape="box", style="filled,rounded", fillcolor="#E2F0D9")
arch.node("llm", "Core LLM Backend\n- Frozen Base Model\n- Inject LoRA Adapter", shape="box", style="filled,rounded", fillcolor="#D9E1F2")
arch.node("post", "Response Post-Processor\n- Final Safety Filters\n- Compliance Logging", shape="box", style="filled,rounded", fillcolor="#FCE4D6")

arch.edge("client", "gateway")
arch.edge("gateway", "orchestrator")
arch.edge("orchestrator", "kb")
arch.edge("orchestrator", "llm")
arch.edge("kb", "llm")
arch.edge("llm", "post")
arch.edge("post", "client")

arch.render("heliex_architecture")

# === 2. Generate LoRA Workflow Diagram ===
lora = Digraph("Heliex_LoRA_Workflow", format="png")
lora.attr(rankdir="LR", size="10")

steps = [
    ("1", "Psychologist Registration\nProfile + Consent"),
    ("2", "Upload Example Dialogues\n(Anonymized Data)"),
    ("3", "Data Preprocessing\n(Clean + Split)"),
    ("4", "LoRA Training Pipeline\n(Parameter-Efficient Fine-Tuning)"),
    ("5", "Validation & Safety QA\n(Clinician + Automated)"),
    ("6", "Store Adapter & Index\n(Persona KB)"),
    ("7", "Persona Activation\n(Available in App)"),
    ("8", "Client Inference\n(LoRA Adapter Loaded)"),
    ("9", "Monitoring & Feedback\n(Retrain if needed)")
]
colors = ["#CFE2F3","#D9EAD3","#FFF2CC","#F9CB9C","#F4CCCC","#D9D2E9","#C9DAF8","#E6E6E6","#E2F0D9"]

for i, (key, label) in enumerate(steps):
    lora.node(key, label, shape="box", style="filled,rounded", fillcolor=colors[i], fontsize="11", fontname="Arial")

for i in range(1, 9):
    lora.edge(str(i), str(i + 1))
lora.edge("9", "3", style="dashed", label="Retrain Loop")

lora.render("heliex_lora_workflow")


# === 3. Convert Diagrams to Base64 ===
def to_base64_image(path):
    with open(path + ".png", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

arch_b64 = to_base64_image("heliex_architecture")
lora_b64 = to_base64_image("heliex_lora_workflow")


# === 4. Build Markdown Content ===
md_content = f"""
# 🧠 Heliex: LLM-Powered Empathic Companion for Mental Health Support  
### *System Architecture and Adaptive Persona Modeling via LoRA Adapters*

---

## 1. Overview
Heliex is a mental-health chatbot designed to simulate the communication style of licensed psychologists.  
It combines a large language model (LLM), prompt-based personas, and optional LoRA adapters for personalization.  
Psychologists register, describe their approach, and optionally upload anonymized dialogues to help model their style.

---

## 2. Functional Goals
- **Psychologist-Driven Personalization**
- **Empathic Conversational Support**
- **Safety and Compliance**
- **Scalable Persona Framework**

---

## 3. Heliex System Architecture

![Heliex System Architecture](data:image/png;base64,{arch_b64})

**Key Layers:**
- Frontend: Web/mobile clients for users and psychologists
- API Gateway: Handles requests and authentication
- Persona Orchestration: Loads LoRA, applies guardrails
- Persona KB: Stores prompt, dialogues, LoRA adapter
- Core LLM: Base model + optional adapter
- Postprocessor: Safety and compliance filters

---

## 4. LoRA Adapter Personalization

**Low-Rank Adaptation (LoRA)** enables lightweight fine-tuning by inserting small trainable matrices into the model.  
Each psychologist’s tone and therapeutic style is encoded in a small adapter (<200MB), loaded dynamically at runtime.

---

### Workflow

![Heliex LoRA Workflow](data:image/png;base64,{lora_b64})

**Steps:**
1. Psychologist Registration  
2. Upload Example Dialogues  
3. Data Preprocessing  
4. LoRA Training  
5. Validation & Safety QA  
6. Store Adapter  
7. Persona Activation  
8. Client Inference  
9. Monitoring & Retraining

---

## 5. Strategic Impact

Heliex bridges the gap between generic conversational AI and authentic psychological empathy.  
It’s scalable, ethical, and adaptable — ensuring every conversation feels genuinely human and clinically safe.

---

**End of Report**
"""

# === 5. Write Markdown File ===
with open("heliex_report.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print("✅ Generated 'heliex_report.md' with embedded diagrams.")
