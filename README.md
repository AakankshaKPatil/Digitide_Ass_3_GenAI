# 💬 Mini-ChatGPT — Hugging Face Text Generator

This is a ChatGPT-like text generation web app built with [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers/).  
It generates text from prompts using local models like GPT-2 — no API key required.

---

## ✨ Features
- Generate AI-based text from any custom prompt.
- Adjustable parameters:
  - **Max tokens**
  - **Temperature** (creativity)
  - **Top-k** and **Top-p** sampling
  - **Number of return sequences**
- Multiple model options: `gpt2`, `distilgpt2`, `gpt2-medium`
- Download generated text as `.txt`
- View session history for current run.

## ⚙ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mini-chatgpt.git
   cd mini-chatgpt
   
2.**(Optional) Create a virtual environment**
  bash
  python -m venv venv
  venv\Scripts\activate      # Windows
  source venv/bin/activate   # Mac/Linux

3.**Install dependencies**
  bash
  pip install -r requirements.txt

4.**Run the app**
  bash
  streamlit run Ass_3.py

5.**Open in your browser**
  arduino
  http://localhost:8501

