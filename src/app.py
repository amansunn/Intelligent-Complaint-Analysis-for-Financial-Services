import gradio as gr
from rag_pipeline import retrieve_relevant_chunks, build_prompt, generate_answer

def rag_chat(question):
    if not question.strip():
        return "Please enter a question.", ""

    # 1. Retrieve chunks
    results = retrieve_relevant_chunks(question, k=5)
    context = [r["text"] for r in results]

    # 2. Build prompt and generate
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)

    # 3. Show 1–2 sources
    sources = "\n\n---\n\n".join(context[:2])

    return answer.strip(), sources

# Gradio Interface
with gr.Blocks(title="CrediTrust RAG Assistant") as demo:
    gr.Markdown("## 🧠 CrediTrust AI Complaint Analyst")
    gr.Markdown("Ask a question based on customer complaint narratives.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your question", placeholder="e.g. What do people complain about BNPL services?", lines=2)
    
    with gr.Row():
        ask_button = gr.Button("Ask")
        clear_button = gr.Button("Clear")

    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
    source_box = gr.Textbox(label="Source Chunks (Top 2)", lines=6, interactive=False)

    ask_button.click(fn=rag_chat, inputs=input_box, outputs=[answer_box, source_box])
    clear_button.click(lambda: ("", "", ""), inputs=None, outputs=[input_box, answer_box, source_box])

# Run the app
if __name__ == "__main__":
    demo.launch()
