import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, google, elevenlabs
load_dotenv(dotenv_path=".env")
def llm_model():
    # function to select the LLM model to use; you will be prompted to select a model (groq, openai or google)
    model_model = input("Which LLM model do you want to use? (groq, openai or google):\n")
    if model_model == "groq":
        llm_obj = openai.LLM.with_groq(model="llama-3.3-70b-versatile")
    elif model_model == "openai":
        llm_obj = openai.LLM.with_openai(model="gpt-3.5-turbo")
    elif model_model == "google":
        llm_obj = google.LLM(model="gemini-2.0-flash-exp", temperature="0.8")
    else:
        llm_obj = openai.LLM.with_openai(model="gpt-3.5-turbo")
    return llm_obj

def stt_model():
    # function to select the STT model to use; you will be prompted to select a model (deepgram, silero, google, groq, or openai)
    model_model = input("Which STT model do you want to use? (deepgram, silero, google, groq or openai):\n")
    if model_model == "deepgram":
        stt_obj = deepgram.STT()
    elif model_model == "silero":
        stt_obj = silero.STT()
    elif model_model == "google":
        stt_obj = google.STT(model="chirp", spoken_punctuation=True)
    elif model_model == "groq":
        stt_obj = openai.STT.with_groq(model="whisper-1")
    elif model_model == "openai":
        stt_obj = openai.STT.with_openai(model="whisper-1")
    else:
        stt_obj = deepgram.STT()
    return stt_obj

def tts_model():
    # function to select the TTS model to use; you will be prompted to select a model (cartesia, google, or elevenlabs)
    model_model = input("Which TTS model do you want to use? (cartesia, google or elevenlabs):\n")
    if model_model == "google":
        tts_obj = google.TTS(gender="female", voice_name="en-US-Standard-H")
    elif model_model == "cartesia":
        tts_obj = cartesia.TTS()
    elif model_model == "elevenlabs":
        tts_obj = elevenlabs.TTS()
    else:
        tts_obj = cartesia.TTS()
    return tts_obj


