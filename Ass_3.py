# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import textwrap

# -------------------------
# Helper: load model once
# -------------------------
@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    # Prefer smaller models for CPU; if you have GPU it'll pick it up when device=0
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return gen

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Mini-ChatGPT (Hugging Face)", layout="centered")

st.title("Mini-ChatGPT — Text Generator (Hugging Face)")
st.markdown(
    "Enter a prompt and click **Generate**. This runs a local Hugging Face model (GPT-2 by default)."
)

# Sidebar: model + info
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Model",
    options=["gpt2", "distilgpt2", "gpt2-medium"],
    index=0,
    help="Select a model. Larger models produce better text but need more RAM/compute."
)

use_gpu = torch.cuda.is_available()
st.sidebar.write("GPU available:" , use_gpu)

# generation params
st.sidebar.subheader("Generation parameters")
max_length = st.sidebar.slider("Max tokens (length)", 50, 800, 150, step=10)
temperature = st.sidebar.slider("Temperature (creativity)", 0.1, 1.5, 0.8, step=0.1)
top_k = st.sidebar.slider("Top-k (0 = disabled)", 0, 200, 50, step=10)
top_p = st.sidebar.slider("Top-p (nucleus)", 0.0, 1.0, 0.95, step=0.05)
num_return_sequences = st.sidebar.slider("Return sequences", 1, 3, 1)

# main prompt area
prompt = st.text_area("Prompt", value="Write a short inspiring paragraph about persistence:", height=150)
col1, col2 = st.columns([1, 1])
with col1:
    seed_input = st.number_input("Random seed (0 = random)", value=0, step=1)
with col2:
    append_prompt = st.checkbox("Append prompt to output (keep original prompt visible)", value=True)

# Load model (cached)
with st.spinner(f"Loading {model_name} (cached after first load)..."):
    generator = load_generator(model_name)

# Generate button
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        # Build generation kwargs
        gen_kwargs = {
            "max_length": max_length,
            "temperature": float(temperature),
            "top_k": int(top_k) if top_k > 0 else None,
            "top_p": float(top_p),
            "num_return_sequences": int(num_return_sequences),
            "do_sample": True if temperature > 0 else False,
        }
        # remove None values to avoid pipeline warnings
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if seed_input != 0:
            torch.manual_seed(seed_input)

        with st.spinner("Generating... this may take a few seconds (model cached)"):
            try:
                outputs = generator(prompt, **gen_kwargs)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

        # show results
        st.subheader("Generated Output")
        results_texts = []
        for i, out in enumerate(outputs):
            text = out["generated_text"]
            # Optionally remove the prefix prompt to show only continuation:
            if not append_prompt:
                if text.startswith(prompt):
                    text = text[len(prompt) :].strip()
            # Nicely wrap long lines for UI
            wrapped = textwrap.fill(text, width=120)
            st.markdown(f"**Result {i+1}:**")
            st.code(wrapped, language="text")
            results_texts.append(text)

        # session history (in-memory)
        if "history" not in st.session_state:
            st.session_state.history = []
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "model": model_name,
            "params": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
                "seed": seed_input,
            },
            "outputs": results_texts,
        }
        st.session_state.history.insert(0, entry)

        # Download button for combined outputs
        combined = "\n\n---\n\n".join(
            [f"Prompt: {prompt}\n\nOutput:\n{t}" for t in results_texts]
        )
        st.download_button(
            label="Download output (.txt)",
            data=combined,
            file_name="generated_text.txt",
            mime="text/plain",
        )

# Show session history
st.markdown("---")
st.subheader("Session history (this session only)")
if "history" in st.session_state and st.session_state.history:
    for idx, h in enumerate(st.session_state.history[:8]):
        st.markdown(f"**{idx+1}.** `{h['timestamp']}` — *{h['model']}*")
        st.markdown(f"> **Prompt:** {h['prompt']}")
        for i, out in enumerate(h["outputs"]):
            st.markdown(f"> **Output {i+1}:** {out[:300]}{'...' if len(out)>300 else ''}")
else:
    st.info("No history yet. Generate some text!")

st.markdown("---")
st.caption("Built with Hugging Face Transformers + Streamlit. Not an official ChatGPT replica — local generation only.")
