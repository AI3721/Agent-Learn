import os
import gradio as gr
import pandas as pd
from chain import chat_chain
from functools import partial

def chat(prompt, chatbot):
    if prompt.strip():
        answer = chat_chain(prompt)
        chatbot.append([prompt, answer])
    return "", chatbot

def save(df: pd.DataFrame, file_path):
    df.to_csv(file_path, index=False)
def create_Tabs(dir="tables"):
    for file in os.listdir(dir):
        file_path = f"{dir}/{file}"
        file_name = file.split('.')[0]
        save_fn = partial(save, file_path=file_path)
        df = pd.read_csv(file_path)
        with gr.Tab(file_name):
            locals()[file_name] = gr.Dataframe(value=df, interactive=True)
            locals()[file_name].change(fn=save_fn, inputs=locals()[file_name])

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; font-size: 2em'>Chat 21</h1>")
    with gr.Row():
        with gr.Column():
            create_Tabs()
        with gr.Column():
            chatbot = gr.Chatbot(label="Agent", avatar_images=['image/ME.png','image/AI.png'])
            prompt = gr.Textbox(label="Prompt", max_lines=1, autofocus=True, interactive=True)
            with gr.Row():
                clear_btn = gr.ClearButton(value="清空", components=[prompt, chatbot])
                submit_btn = gr.Button(value="提交", variant='primary', interactive=True)
            submit_btn.click(fn=chat, inputs=[prompt, chatbot], outputs=[prompt, chatbot])
            prompt.submit(fn=chat, inputs=[prompt, chatbot], outputs=[prompt, chatbot])

demo.launch(inbrowser=True, share=False)