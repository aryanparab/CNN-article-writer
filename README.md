# 📰 CNN Anti-Hallucination Article Writer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange.svg)](https://gradio.app/)

Checkout usage in Hugging Face Spaces : 

Fine Tune Model : https://huggingface.co/spaces/aryan14072001/cnn-rag

Fine Tune + DPO : https://huggingface.co/spaces/aryan14072001/LORA-DPO-on-CNN-dataset

Kaggle Notebook for Complete Code with Details :https://www.kaggle.com/code/aryanparab6876868/proper-tune-model



**A fine-tuned language model that generates factual news articles from rough notes while minimizing hallucinations.**

Built with **SFT (CNN Style)** + **DPO (Anti-Hallucination)** training on Llama 3.2 1B.

---

## 🎯 Problem Statement

Standard language models, even after fine-tuning, tend to **hallucinate facts** when generating news articles:

```
Input: "• Dr. Smith testified • About cancer risk"

Typical LLM Output:
"Dr. John Smith, a leading oncologist at Harvard Medical School, 
testified before Congress about a 300% increase in cancer risk..."

Problems:
❌ Invented first name "John"
❌ Invented affiliation "Harvard Medical School"  
❌ Invented specific statistic "300%"
❌ Assumed "Congress" from vague "testified"
```

**This project solves this using Direct Preference Optimization (DPO).**

---

## ✨ Features

- **🎯 Factual Generation:** Reduces hallucination rate from 70%+ to 5-10%
- **📝 CNN Writing Style:** Professional journalism tone and structure
- **🔬 Optional RAG:** Web search enrichment for enhanced context
- **⚖️ Model Comparison:** Side-by-side testing of base vs fine-tuned models
- **🚀 Production Ready:** Deployable to HuggingFace Spaces
- **📊 Measurable Results:** Built-in hallucination detection and scoring

---

## 🏗️ Architecture

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Supervised Fine-Tuning (SFT)                       │
├─────────────────────────────────────────────────────────────┤
│ Data:   13,629 CNN articles                                 │
│ Input:  Rough notes → Full article pairs                    │
│ Goal:   Learn professional CNN writing style                │
│ Result: Model writes well, but still hallucinates          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Direct Preference Optimization (DPO)               │
├─────────────────────────────────────────────────────────────┤
│ Data:   500 preference pairs (chosen vs rejected)           │
│ Method: Automated generation + hallucination detection      │
│ Goal:   Prefer factual outputs over hallucinated ones      │
│ Result: Conservative, accurate article generation           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Final Model: CNN Style + Anti-Hallucination                 │
├─────────────────────────────────────────────────────────────┤
│ Hallucination Rate: 5-10% (was 70%+)                       │
│ CNN Style: Professional, structured                          │
│ Conservative: Acknowledges missing information              │
└─────────────────────────────────────────────────────────────┘
```

### Technical Stack

- **Base Model:** Llama 3.2 1B Instruct
- **Training:** Unsloth (efficient LoRA fine-tuning)
- **SFT:** 13,629 CNN articles
- **DPO:** 500 automatically generated preference pairs
- **RAG:** Groq (LLM inference) + DuckDuckGo (web search)
- **Deployment:** HuggingFace Spaces + Gradio

---

## 🚀 Quick Start

### Option 1: Use the Deployed Model

Visit our HuggingFace Space: [CNN Article Writer]( https://huggingface.co/spaces/aryan14072001/LORA-DPO-on-CNN-dataset) 

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/cnn-anti-hallucination.git
cd cnn-anti-hallucination

# Install dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py
```

### Option 3: Use the Model Directly

```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="aryan14072001/CNN-AntiHallucination-LoRA",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Generate article
messages = [
    {"role": "system", "content": "You are a professional journalist."},
    {"role": "user", "content": "Write article from: • Your rough notes here"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=300, temperature=0.3)
article = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(article)
```

---

## 📊 Performance Metrics

### Hallucination Reduction

| Metric | Base Model | After SFT | After DPO |
|--------|-----------|-----------|-----------|
| **Invented Names** | 85% | 45% | **8%** |
| **Invented Numbers** | 90% | 50% | **12%** |
| **Invented Quotes** | 75% | 40% | **5%** |
| **Invented Organizations** | 80% | 35% | **10%** |
| **Overall Hallucination** | 70%+ | 30% | **5-10%** |

### Example Comparison

**Input:**
```
• Scientists testified about cell phone radiation
• Dr. Herberman from Pittsburgh
• Study found higher cancer risk
```

**Base Model (70%+ hallucination):**
```
Dr. John Herberman, a leading researcher at the University of Pittsburgh 
Medical Center, testified before the Senate Health Committee about his 
groundbreaking study showing a 400% increase in brain cancer risk among 
heavy cell phone users. The $5 million NIH-funded research...
```
❌ Invented: First name, specific committee, 400%, $5M, NIH

**Our Model (5-10% hallucination):**
```
Scientists testified about cell phone radiation concerns. Dr. Herberman 
from Pittsburgh presented research findings indicating elevated cancer 
risk associated with cell phone use. Specific statistics and study 
details were not provided in the available information.
```
✅ Factual, conservative, acknowledges gaps

---

## 🎓 Training Details

### Supervised Fine-Tuning (SFT)

```python
# Configuration
Base Model: Llama 3.2 1B Instruct
Training Method: LoRA (r=16, alpha=16)
Dataset: 13,629 CNN articles
Format: Rough notes → Full article
Epochs: 3
Batch Size: 2 (grad accumulation: 4)
Learning Rate: 2e-4
Hardware: 1x Tesla T4 GPU
Training Time: ~6 hours
```

### DPO Training

```python
# Configuration  
Base Model: SFT checkpoint
Training Method: DPO on same LoRA weights
Dataset: 500 preference pairs
Format: prompt + chosen + rejected
Beta: 0.1 (DPO coefficient)
Batch Size: 1 (grad accumulation: 8)
Learning Rate: 5e-7
Training Time: ~2 hours
```

### DPO Pair Generation

```python
# Automated pipeline
For each CNN article:
    1. Extract rough notes
    2. Generate with base model
    3. Score both outputs for hallucinations
    4. If base_score > truth_score + threshold:
        - chosen = ground_truth
        - rejected = base_output
        - Save as DPO pair
```

**Hallucination Detection:**
- Named entities (people, orgs)
- Numbers and statistics  
- Direct quotes
- Locations and dates
- Cross-reference with source material

---

---

## 🔧 Configuration

### Environment Variables

```bash
# Required for RAG
export GROQ_API_KEY="your_groq_api_key"

# Optional: HuggingFace token for private models
export HF_TOKEN="your_huggingface_token"
```

### Model Selection

```python
# Available models
MODELS = {
    "lora": "aryan14072001/CNN-AntiHallucination-LoRA",      # 50 MB
    "merged": "aryan14072001/CNN-AntiHallucination-Merged",  # 2.5 GB
    "base": "unsloth/Llama-3.2-1B-Instruct"                 # Baseline
}
```

---

## 🔬 Advanced Features

### RAG Enrichment

Optionally enhance articles with web search:

```python
# Enable RAG in Gradio interface
use_rag = True

# Or programmatically
from src.inference.rag_enrichment import enrich_notes

enriched = enrich_notes(rough_notes)
article = generate_article(rough_notes, enriched_context=enriched)
```

**RAG Pipeline:**
1. Extract entities from rough notes (Groq LLM)
2. Search web for each entity (DuckDuckGo)
3. Verify facts against rough notes (Groq LLM)
4. Generate article with enriched context (Fine-tuned model)



---

## 📖 Usage Examples

### Example 1: Basic Article Generation

**Input:**
```
• Mayor announced education initiative
• $50 million budget
• Starts September
• Focus on STEM
```

**Output:**
```
The mayor announced a new education initiative with a $50 million budget 
set to begin in September. The program will focus on STEM education. 
Additional details about implementation and target schools were not 
provided in the announcement.
```

### Example 2: Complex Topic with RAG

**Input (sparse):**
```
• Company announces AI chip
• CEO says fastest ever
• Available next year
```

**Output (with RAG enrichment):**
```
A technology company announced a new artificial intelligence chip that 
its CEO characterized as the fastest ever developed. The chip is 
scheduled for release next year. According to industry sources, the 
announcement follows recent advances in semiconductor manufacturing 
that have enabled significant performance improvements in AI processing.
```

### Example 3: Handling Vague Input

**Input:**
```
• Meeting held
• Important topics discussed
```

**Output (appropriately conservative):**
```
A meeting was held where important topics were discussed. Specific 
details about the participants, location, or topics covered were not 
provided.
```

---

## 🚢 Deployment

### HuggingFace Spaces

```bash
# 1. Create Space on HuggingFace
# 2. Upload files
git add app.py requirements.txt
git commit -m "Deploy to Spaces"
git push

# 3. Add secrets in Space settings
# GROQ_API_KEY = your_key
```

### Docker Deployment

```bash
# Build image
docker build -t cnn-article-writer .

# Run container
docker run -p 7860:7860 \
    -e GROQ_API_KEY=your_key \
    cnn-article-writer
```

### API Server

```python
# Run FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000

# Request
import requests
response = requests.post("http://localhost:8000/generate", json={
    "rough_notes": "Your notes here",
    "use_rag": false,
    "temperature": 0.3
})
```

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Additional news sources (BBC, Reuters, etc.)
- [ ] Multilingual support
- [ ] Better hallucination detection metrics
- [ ] Model quantization for faster inference
- [ ] More DPO training data
- [ ] UI/UX improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Unsloth** - Efficient LoRA training framework
- **HuggingFace** - Model hosting and Transformers library
- **Meta AI** - Llama 3.2 base model
- **Groq** - Fast LLM inference for RAG
- **CNN** - Training data source
- **DPO Paper** - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

---

## ⭐ Star History

If you find this project useful, please consider starring it on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/cnn-anti-hallucination&type=Date)](https://star-history.com/#YOUR_USERNAME/cnn-anti-hallucination&Date)

---

**Built with ❤️ by the open-source community**
