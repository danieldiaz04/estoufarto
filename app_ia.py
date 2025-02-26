import streamlit as st
import openai
from PIL import Image
import io
import base64
import os
import time
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Criar um assistente (executar apenas uma vez e guardar o ID)
assistant = openai.beta.assistants.create(
    name="Diagnóstico Automóvel",
    instructions="És um mecânico automotivo especializado em diagnóstico de problemas.",
    model="gpt-4-turbo"
)
ASSISTANT_ID = assistant.id

# Função para criar um novo thread

def create_thread():
    thread = openai.beta.threads.create()
    return thread.id

# Função para converter imagem para base64
def encode_image(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# Função para transcrever áudio
def transcribe_audio(audio_file):
    audio_bytes = audio_file.read()
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=io.BytesIO(audio_bytes),
        response_format="text"
    )
    return response

st.title("Diagnóstico Automóvel com IA")
st.write("Descreva o problema do seu carro ou envie um áudio e imagens para análise.")

# Criar um thread único por execução da app
THREAD_ID = create_thread()

# Input de texto
description = st.text_area("Descreva o problema do seu carro:")

# Upload de áudio
uploaded_audio = st.file_uploader("Ou carregue um áudio descrevendo o problema", type=["mp3", "wav", "m4a"])
if uploaded_audio:
    description = transcribe_audio(uploaded_audio)
    st.write("Texto transcrito:", description)

# Upload das imagens
uploaded_panel = st.file_uploader("Carregue uma imagem do painel de instrumentos", type=["jpg", "png", "jpeg"])
uploaded_hood = st.file_uploader("Carregue uma imagem do motor (capô aberto)", type=["jpg", "png", "jpeg"])

if st.button("Analisar Problema"):
    if not description:
        st.error("Por favor, insira uma descrição do problema ou envie um áudio.")
    elif not uploaded_panel or not uploaded_hood:
        st.error("Por favor, carregue ambas as imagens.")
    else:
        # Converter imagens para Base64
        panel_image = encode_image(Image.open(uploaded_panel))
        hood_image = encode_image(Image.open(uploaded_hood))
        
        # Criar mensagem no thread
        openai.beta.threads.messages.create(
            thread_id=THREAD_ID,
            role="user",
            content=f"""
            O utilizador descreveu o seguinte problema com o carro:
            {description}
            
            Analise as imagens do painel de instrumentos e do motor e forneça um diagnóstico provável, incluindo possíveis causas e soluções recomendadas.
            """
        )

        # Criar execução do assistente
        run = openai.beta.threads.runs.create(
            thread_id=THREAD_ID,
            assistant_id=ASSISTANT_ID
        )
        
        # Aguardar resposta do assistente
        while run.status not in ["completed", "failed"]:
            time.sleep(2)
            run = openai.beta.threads.runs.retrieve(run.id)
        
        # Obter resposta
        messages = openai.beta.threads.messages.list(thread_id=THREAD_ID)
        st.subheader("Diagnóstico do Problema")
        st.write(messages.data[0].content[0].text.value)
