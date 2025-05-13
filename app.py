import streamlit as st
import os
import pytube
import yt_dlp
import moviepy.editor as mp
from pydub import AudioSegment
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import ffmpeg
import time
import logging

# Set ffmpeg path explicitly
os.environ["IMAGEIO_FFMPEG_EXE"] = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg\\tools\\ffmpeg\\bin\\ffmpeg.exe"
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

# Streamlit app layout
st.title("Video Processing and Question Answering App")
st.write("Upload a video file or provide a YouTube link to extract audio, transcribe it, and ask questions based on the content.")

# Video input
input_option = st.radio("Choose input method:", ("Upload Video File", "YouTube URL"))
video_file = None
youtube_url = None

if input_option == "Upload Video File":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
else:
    youtube_url = st.text_input("Enter YouTube URL")

# Process button
if st.button("Process Video"):
    if not video_file and not youtube_url:
        st.error("Please provide a video file or YouTube URL.")
    else:
        try:
            # Reset session state
            st.session_state.transcription = None
            st.session_state.video_path = None
            st.session_state.processing_status = "Starting..."

            # Step 1: Save or download video
            st.session_state.processing_status = "Saving/Downloading video..."
            st.info(st.session_state.processing_status)
            if video_file:
                video_path = os.path.join("temp", video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.getbuffer())
                st.session_state.video_path = video_path
            else:
                # Download YouTube video using yt-dlp
                ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',
                    'outtmpl': 'temp/youtube_video.%(ext)s',
                    'merge_output_format': 'mp4',
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    video_path = ydl.prepare_filename(info)
                    # Ensure the extension is mp4
                    if not video_path.endswith(".mp4"):
                        video_path = video_path.rsplit(".", 1)[0] + ".mp4"
                    st.session_state.video_path = video_path

            # Step 2: Extract audio
            st.session_state.processing_status = "Extracting audio..."
            st.info(st.session_state.processing_status)
            video = mp.VideoFileClip(video_path)
            audio_path = os.path.join("temp", "audio.wav")
            video.audio.write_audiofile(audio_path)
            video.close()

            # Step 3: Transcribe audio using Whisper
            st.session_state.processing_status = "Transcribing audio..."
            st.info(st.session_state.processing_status)
            model_id = "openai/whisper-small"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=0 if device == "cuda" else -1,
            )
            audio = AudioSegment.from_wav(audio_path)
            # Log audio duration for debugging
            logger.info(f"Audio duration: {len(audio) / 1000} seconds")
            # Convert to mono and resample if needed
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(audio_path, format="wav")
            # Add return_timestamps=True to handle long-form audio
            transcription = pipe(audio_path, generate_kwargs={"language": "english", "return_timestamps": True})
            st.session_state.transcription = transcription["text"]
            st.session_state.processing_status = "Processing complete!"
            st.success(st.session_state.processing_status)

            # Clean up temporary files
            try:
                os.remove(audio_path)
                if video_file:
                    os.remove(video_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")

        except Exception as e:
            st.session_state.processing_status = f"Error: {str(e)}"
            st.error(st.session_state.processing_status)
            logger.error(f"Processing failed: {e}")

# Display video and transcription
if st.session_state.video_path and os.path.exists(st.session_state.video_path):
    st.subheader("Watch Video")
    st.video(st.session_state.video_path)

if st.session_state.transcription:
    st.subheader("Transcription")
    st.write(st.session_state.transcription)

    # Question-Answering section
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question about the video content")
    if question:
        try:
            st.info("Processing your question...")
            qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
            answer = qa_pipeline(question=question, context=st.session_state.transcription)
            st.write(f"Answer: {answer['answer']} (Confidence: {answer['score']:.2f})")
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
            logger.error(f"QA failed: {e}")

# Display current status
if st.session_state.processing_status:
    st.write(f"Status: {st.session_state.processing_status}")
