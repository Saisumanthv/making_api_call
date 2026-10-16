import json
import requests
import streamlit as st

st.title("Local LLM Chat (Ollama)")

# Model selection and prompt input
model_name = "llama3.2"
prompt = st.text_area("Enter your prompt:", "Who is Elon Musk?")

if st.button("Generate Response", type="primary"):
  if not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt}


    # Generator function to yield chunks from Ollama's stream
    def generate_tokens():
      with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
          if line:
            chunk = json.loads(line)
            yield chunk.get("response", "")
            if chunk.get("done"):
              break


    # Streamlit renders the generator output in real time
    st.write_stream(generate_tokens())