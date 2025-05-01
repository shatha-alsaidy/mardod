 
import platform
from pathlib import Path
import subprocess
import numpy as np

# FastAPI core imports:
# FastAPI: for building the web API``
# HTTPException: to raise API errors (e.g. 400, 500 responses)
# BackgroundTasks: to run tasks in the background without blocking the request
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
# Typing hints for clarity and IDE support
from typing import List, Optional, Dict, Any , Union
# Standard Python utilities:
import re                    # Regular expressions for text cleaning
import os                    # For interacting with the operating system (file paths, etc.)
import logging               # For logging messages (info, warnings, errors)
import tempfile              # For creating temporary files/folders during processing
import concurrent.futures    # For running tasks concurrently (multi-threaded)
import uuid 

               # For generating unique identifiers for documents/items
#
# Whisper - OpenAI's speech-to-text model (for transcribing audio)
import whisper
 
# Chroma vector store used to store and search document embeddings
from langchain_community.vectorstores import Chroma

# ChatOpenAI is a wrapper to interact with GPT models via LangChain
from langchain_openai import ChatOpenAI

# Document class used to represent text chunks with optional metadata
from langchain.schema import Document

# Direct import of ChromaDB client to manage the database
import chromadb

# Utility to generate text embeddings using SentenceTransformers
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# s long text into smaller pieces to improve semantic search accuracy
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Apify SDK client used to run scraping actors for social media (Twitter, TikTok)
from apify_client import ApifyClient

# yt_dlp is used to download audio/video from TikTok and YouTube
import yt_dlp as youtube_dlp

# Direct OpenAI SDK used for moderation and other backend tasks
from openai import OpenAI

# pydantic is used to define structured data models for input/output
from pydantic import BaseModel

# pydantic-settings allows loading .env variables into Python classes easily
from pydantic_settings import BaseSettings  

# Firebase Admin SDK - used to interact with Firebase Realtime Database
import firebase_admin
from firebase_admin import credentials, db

# Used to record and manipulate timestamps (e.g. when tracking keywords)
from datetime import datetime

import os
import shutil

import asyncio
from concurrent.futures import ProcessPoolExecutor

import random
import binascii



# Initialize logging
# Configures the logging system to show messages at INFO level and above

logging.basicConfig(
    format="%(asctime)s %(threadName)s %(levelname)s:%(message)s",
    level=logging.INFO
)

# Creates a logger instance for this file/module
logger = logging.getLogger(__name__)

import functools, inspect

def log_entry_exit(func):
    """Decorator to log entry and exit of both sync and async functions."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.info(f"→ Entering {func.__name__}")
            result = await func(*args, **kwargs)
            logger.info(f"← Exiting {func.__name__}")
            return result
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.info(f"→ Entering {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"← Exiting {func.__name__}")
            return result
        return sync_wrapper


# =============================================================================
# Configuration and Models Setup
# =============================================================================
# This class loads project-wide configuration from environment variables (.env file)
class Settings(BaseSettings):
    openai_api_key: str               # API key for OpenAI services (chat, moderation)
    apify_api_key: str                # API key for Apify platform (web scraping)
    whisper_model_size: str           # Size of Whisper model to load (e.g., "base", "small")
    chroma_path: str                  # Directory path where Chroma vector DB is stored
    embedding_model: str              # Name of the sentence transformer model for embeddings
    collection_name: str              # Name of the vector collection in ChromaDB
    max_resultsx: int                 # Max number of Twitter results to fetch per keyword
    max_resultstiktok: int            # Max number of TikTok videos to fetch per keyword
    max_keywords: int                 # Max number of keywords to extract from the query
    similarity_threshold: float       # Similarity threshold for filtering search results

    # Firebase keys (commented for now, can be used later if needed)
    # firebase_cred_path: str
    # firebase_db_url: str

    class Config:
        env_file = ".env"             # Tells Pydantic to load variables from a `.env` file

# Instantiate and load all config into a single `settings` object
settings = Settings()

# src = the folder in your Git repo containing the pre-collected data
src = os.path.join(os.getcwd(), "chroma_db")  

# mount = the Render disk mount path
mount = settings.chroma_path             

# make an app-owned subfolder
dest = os.path.join(mount, "chroma_db")   # <-- new

# only copy on first run (when dest is empty/nonexistent)
if os.path.isdir(src) and not os.path.exists(dest):
    os.makedirs(dest, exist_ok=True)      # ensure parent → dest exists
    shutil.copytree(src, dest, dirs_exist_ok=True)


# Initialize components at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and resources"""
    try:
        logger.info("Initializing OpenAI client...")
        app.state.openai_client = OpenAI(api_key=settings.openai_api_key)

        logger.info("Loading Whisper model...")
        app.state.whisper_model = whisper.load_model(settings.whisper_model_size, device='cpu',download_root="./models")
        
        logger.info("Initializing Keyword Extraction model...")
        app.state.kw_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )

        logger.info("Initializing Chat model...")
        app.state.chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )

        logger.info("Initializing ChromaDB...")
        app.state.chroma_client = chromadb.PersistentClient(path=dest)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        app.state.collection = app.state.chroma_client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn
        )

        '''logger.info("Initializing Firebase...")
        cred = credentials.Certificate(settings.firebase_cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': settings.firebase_db_url
        })
        app.state.firebase_ref = db.reference('/trending_keywords')'''

        logger.info("Initialization complete")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise
    yield
    # Shutdown logic here
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)


# =============================================================================
# Pydantic Models
# =============================================================================
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# =============================================================================
# Core Processing Functions (Concurrent)
# =============================================================================
@log_entry_exit
async def process_platform_items(items: List[Dict[str, Any]],platform: str,background_tasks: BackgroundTasks) -> List[Document]:
    """Process items with proper async handling and result processing"""
    try:
        if platform == 'tiktok':
            # Sequential processing for TikTok
            documents = []
            for item in items:
                result = await asyncio.to_thread(
                    process_single_item,
                    item=item,
                    platform=platform,
                    background_tasks=background_tasks
                )
                if result:
                    documents.extend(result)
                # Force memory cleanup
                import gc
                gc.collect()
            return documents
        else:
            # Twitter processing with limited parallelism
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        executor,
                        functools.partial(
                            process_single_item,
                            item=item,
                            platform=platform,
                            background_tasks=background_tasks
                        )
                    )
                    for item in items
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Direct result processing (replaces process_results)
                documents = []
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Processing error: {str(result)}")
                    elif result:
                        documents.extend(result)
                
                return documents
    except Exception as e:
        logger.error(f"Platform processing failed: {str(e)}")
        return []
    
@log_entry_exit
def process_single_item(item: Dict[str, Any],platform: str,background_tasks: BackgroundTasks) -> List[Document]:
    """Process a single social media item into chunks"""
    try:
        # Common processing
        doc_uuid = str(uuid.uuid4())
        timestamp = item.get("createdAt") or item.get("createTimeISO", "")
        url = item.get("url") or item.get("webVideoUrl", "")
        
        # Platform-specific processing
        if platform == 'tiktok':
            text = process_tiktok_item(item, background_tasks)
        elif platform == 'twitter':
            text = clean_text(item.get("text", ""))
        else:
            return []

        # Chunking and document creation
        chunks = chunk_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            if is_helpful(chunk):
                print(chunk)
                documents.append({
                    "id": f"{doc_uuid}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "source": platform,
                        "url": url,
                        "timestamp": timestamp,
                        "doc_uuid": doc_uuid,
                        "chunk_num": i+1,
                        "total_chunks": len(chunks)
                    }
                })
        
        return documents
    except Exception as e:
        logger.error(f"Processing document failed: {str(e)}")
        return []
@log_entry_exit
def process_tiktok_item(item: Dict[str, Any], background_tasks: BackgroundTasks) -> str:
    """Process TikTok without temp files"""
    try:
        video_url = item.get("webVideoUrl")
        if not video_url:
            return ""
        
        # Stream audio directly
        audio_url = download_audio(video_url)
        if not audio_url:
            return ""
        
        # Transcribe from URL
        transcription = transcribe_audio(audio_url)
        return clean_text(f"{item.get('text', '')} {transcription}")
    except Exception as e:
        logger.error(f"TikTok processing failed: {str(e)}")
        return ""

 
@log_entry_exit
def cleanup_temp_files(file_path: str):
    """Cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

# =============================================================================
# Helper Functions
# =============================================================================
@log_entry_exit
def chunk_text(text: str) -> List[str]:
    """Text chunking with optimized parameters"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)
@log_entry_exit
def clean_text(text: str) -> str:
    """Clean text from emojis and unnecessary characters"""
    # Emoji pattern
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"  # Supplementary Multilingual Plane (most emojis)
        "|\U0000ff00-\U0000ffff"  # Halfwidth and Fullwidth Forms (includes some emojis)
        "|\U00002000-\U000027bf"  # Dingbats and other symbols
        "|\U00002400-\U00002bff"  # Additional symbols and pictographs
        "|\U0001f300-\U0001f5ff"  # Miscellaneous Symbols and Pictographs
        "|\U0001f600-\U0001f64f"  # Emoticons
        "|\U0001f680-\U0001f6ff"  # Transport and Map symbols
        "|\U0001f700-\U0001f77f"  # Alchemical symbols
        "|\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "|\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "|\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "|\U0001fa00-\U0001fa6f"  # Chess Symbols
        "|\U0001f000-\U0001f02f"  # Mahjong Tiles
        "|\U0001f0a0-\U0001f0ff"  # Playing Cards
        "|\U0001f100-\U0001f1ff"  # Enclosed Alphanumeric Supplement
        "|\U0001f200-\U0001f251"  # Enclosed Ideographic Supplement
        "|\U0001f252-\U0001f25f"  # Enclosed Ideographic Supplement
        "|\U00002b50]"  # Additional symbol(s)
    )
    
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
@log_entry_exit
def is_helpful(text: str) -> bool:
    """Quality check for useful content"""
    # Basic length check
    words = text.split()
    if len(words) < 5:  
        return False

    # Check for question-like patterns (Arabic question mark)
    if '؟' in text:
        # Count question marks and check position
        q_count = text.count('؟')
        last_char = text.strip()[-1] if text.strip() else ''
        
        # More than 2 question marks or ends with question mark
        if q_count > 2 or last_char == '؟':
            return False

    # Calculate text statistics
    unique_words = len(set(words))
    total_chars = len(text)
    punctuation_count = sum(1 for c in text if c in ('!', '؟', '.', ','))
    avg_word_length = sum(len(word) for word in words) / len(words)

    # Unique word ratio check (avoid repetitive content)
    if unique_words / len(words) < 0.4:  # 55% unique words minimum
        return False

    # Punctuation density check (max 15% punctuation)
    if punctuation_count / total_chars > 0.25:
        return False

    # Average word length check 
    if not (3 <= avg_word_length <= 15):
        return False

    # URL check (simple regex pattern)
    if re.search(r'(https?://|www\.)\S+', text):
        return False

    # Keyword stuffing check (repeated phrases)
    if re.search(r'\b(\w+)(\s+\1){2,}\b', text):
        return False

    # Stopword ratio check (Saudi Arabic-specific)
    arabic_stopwords = { "وش", "ليش", "ايش", "ليه", "كيف", "متى", "وين", "مرة", "ترا", "بس", "كذا", "يعني", "لسه", "لحد",
    "لانه", "عشان", "على", "عن", "في", "من", "ما", "مو", "مش", "ولا", "حتى", "كل", "بعض",
    "هذا", "هذي", "ذي", "ذا", "ذاك", "هذولا", "اللي", "كان", "تكون", "يكون", "تكونوا", "انا",
    "انت", "انتي", "انتم", "احنا", "هم", "هما", "هو", "هي", "ذا", "ده", "دي", "ذا", "ماحد",
    "محد", "ابد", "دايم", "مره", "توني", "تراك", "تدرون", "تدرين", "والله", "بالله", "طيب",
    "اجل", "عاد", "اوكي", "خلاص", "زين", "او", "ولا", "اما", "بس", "بعدين", "شوي", "كثير",
    "مرره", "بكره", "امس", "الحين", "قبل", "بعدها", "لسا","لسه", "ان", "اذا", "لو", "يمكن", "لعل",
    "لان", "ثم", "حين", "حينها", "يالله", "ان شاء الله", "هلا", "يلا", "ياخي","تراك"}

    stopword_count = sum(1 for word in words if word in arabic_stopwords)
    if stopword_count / len(words) > 0.5:  # Max 50% stopwords
        return False

    return True
###
logger.addHandler(logging.NullHandler())

# -------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------
# Cell 3: Logger setup and FFMPEG path
FFMPEG_PATH = (
    Path(r"C:\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin")
    if platform.system() == "Windows"
    else Path("/usr/bin")
)

# Cell 4: download_audio function
@log_entry_exit
def download_audio(url: str, temp_dir: Union[str, Path]) -> Optional[Path]:
    """
    Download audio from a video URL, convert to MP3, and return the Path
    """
    outdir = Path(temp_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    opts = opts = {
        "format": "bestaudio/best",
        "postprocessors": ["-acodec", "libmp3lame"],
        "headers": {
            "User-Agent": "com.zhiliaoapp.musically/2024701030 (Linux; U; Android 14; en_GB; SM-G991B; Build/TP1A.220624.014;tt-ok/3.14.2.3)",
            "Referer": "https://www.tiktok.com/",
        },
        "extractor_args": {
            "tiktok": {
                "app_version": "34.5.3",
                "manifest_app_version": "2024701030",
                "player_client": "android_hybrid"
            }
        },
        "cookiefile": "cookies.txt",
        "sleep_interval": 5,  # Add delay between requests
        "retries": 3,
        "quiet": True,
        "no_warnings": True,
        "verbose": True,  # -v
        "update": True, 
        "outtmpl": "-"
    }

    # Add mobile device parameters
    opts["extractor_args"]["tiktok"].update({
        "device_id": str(random.randint(1000000000000000000, 9999999999999999999)),
        "iid": str(random.randint(7000000000000000000, 7999999999999999999)),
        "openudid": binascii.hexlify(os.urandom(8)).decode(),
    })

    try:
        with youtube_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('url')  # Return direct audio URL
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return None

# Cell 5: strip_silence function
@log_entry_exit
def strip_silence(input_path: Path, output_path: Path) -> bool:
    """
    Trim leading/trailing silence with ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB:detection=peak,areverse,"
               "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB:detection=peak,areverse",
        "-acodec", "mp3", str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.warning("strip_silence failed: %s", e)
        return False

# Cell 6: safe_cleanup function
@log_entry_exit
def safe_cleanup(path: Optional[Union[str, Path]]) -> None:
    """
    Delete the file at `path` if it exists.
    """
    if not path:
        return
    p = Path(path)
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        logger.error("safe_cleanup failed: %s", e)

# Cell 7: preview function
@log_entry_exit
def preview(obj):
    """
    Print a truncated preview of a string or list of (doc, meta) tuples.
    """
    if isinstance(obj, list):
        for i, (doc, _) in enumerate(obj[:5], 1):
            print(f"[{i}] {doc[:120]}…")
    elif isinstance(obj, str):
        print(obj[:500] + "…")
    else:
        print(f"(unexpected type: {type(obj)})")

# Cell 8: transcribe_audio function
@log_entry_exit
def transcribe_audio(audio_url: str) -> Optional[str]:
    """Transcribe directly from URL without temp files"""
    try:
        # Stream audio directly
        audio = whisper.load_audio(audio_url, duration=90)
        audio = whisper.pad_or_trim(audio)
        
        # Use tiny model
        model = app.state.whisper_model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Fast decoding
        options = whisper.DecodingOptions(language="ar", beam_size=1, fp16=False)
        result = whisper.decode(model, mel, options)
        return result.text
    except Exception as e:
        logger.error(f"Streaming transcription failed: {str(e)}")
        return None
    finally:
        import gc
        gc.collect()
#####################

# =============================================================================
# Main Endpoint with Moderation and Validation
# =============================================================================
@log_entry_exit
@app.post("/query", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Main query processing endpoint"""
    try:
        # Moderation check
        if not await asyncio.to_thread(moderate_query, request.query):
            raise HTTPException(400, "Query violates content policies")
        
        # =====================================================================
        # Phase 1: Initial Database Check
        # =====================================================================
        
        # Check existing documents in ChromaDB
        existing_results = await asyncio.to_thread(
            app.state.collection.query,
            query_texts=[request.query],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )

        # Filter documents based on similarity threshold
        threshold_similarity = settings.similarity_threshold 
        filtered_docs = []
        for doc, meta, distance in zip(existing_results['documents'][0], 
                                     existing_results['metadatas'][0], 
                                     existing_results['distances'][0]):
            print(doc+"  ----99  dis: "+str(distance))
            if (1 - distance) >= threshold_similarity and is_helpful(doc): 
                filtered_docs.append((doc, meta))
        
        # If sufficient relevant documents found, return early
        if len(filtered_docs) >= 4:
           # print(filtered_docs.split("999"))
            processed_results = {
                "documents": [[doc for doc, meta in filtered_docs]],
                "metadatas": [[meta for doc, meta in filtered_docs]]
            }
            return generate_response(processed_results, request.query)
        
        # =====================================================================
        # Phase 2: Social Media Search and Processing
        # =====================================================================

        # Keyword extraction
        keywords = await extract_keywords(request.query)
        logger.info(f"Extracted keywords: {keywords}")
        
        if not keywords:
            logger.warning("No valid keywords extracted")
            return fallback_response()

        apify_client = ApifyClient(settings.apify_api_key)
        content = await search_social_media(keywords, apify_client)
        
        # Process content concurrently
        twitter_items = [item for platform, item in content if platform == 'twitter']
        tiktok_items = [item for platform, item in content if platform == 'tiktok']
        
        # Process platforms concurrently
        twitter_docs, tiktok_docs = await asyncio.gather(
            process_platform_items(twitter_items, 'twitter', background_tasks),
            process_platform_items(tiktok_items, 'tiktok', background_tasks),
            return_exceptions=True  # Add this to prevent propagation
        )

        # Handle potential exceptions in platform processing
        if isinstance(twitter_docs, Exception):
            logger.error(f"Twitter processing failed: {twitter_docs}")
            twitter_docs = []
        if isinstance(tiktok_docs, Exception):
            logger.error(f"TikTok processing failed: {tiktok_docs}")
            tiktok_docs = []

        all_documents = twitter_docs + tiktok_docs

        print(all_documents)

        # Add to ChromaDB
        if all_documents:
            await asyncio.to_thread(
                app.state.collection.add,
                documents=[doc["text"] for doc in all_documents],
                metadatas=[doc["metadata"] for doc in all_documents],
                ids=[doc["id"] for doc in all_documents]
            )

        # =====================================================================
        # Phase 3: Secondary Database Check
        # =====================================================================   

        # Check updated documents in ChromaDB
        updated_results = await asyncio.to_thread(
            app.state.collection.query,
            query_texts=[request.query],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )
        
        # Filter documents based on similarity threshold
        filtered_docs = []
        for doc, meta, distance in zip(updated_results['documents'][0], 
                                     updated_results['metadatas'][0], 
                                     updated_results['distances'][0]):
            print(doc+"  ----88  dis: "+str(distance))
            if (1 - distance) >= threshold_similarity and is_helpful(doc): 
                filtered_docs.append((doc, meta))
        
        # If sufficient relevant documents found, return response
        if len(filtered_docs) >= 4:
            processed_results = {
                "documents": [[doc for doc, meta in filtered_docs]],
                "metadatas": [[meta for doc, meta in filtered_docs]]
            }
            return generate_response(processed_results, request.query)
        
        # =====================================================================
        # Fallback: No Relevant Results Found
        # =====================================================================
        logger.warning("No relevant results found after full processing")
        return QueryResponse(
            answer="عذرًا، لم نتمكن من العثور على معلومات مؤكدة حول هذا الموضوع. يرجى المحاولة مرة أخرى لاحقًا أو تقديم تفاصيل أكثر تحديدًا.",
            sources=[]
        )
        

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
def fallback_response():
    return QueryResponse(
        answer="عذرًا، لم نتمكن من العثور على معلومات مؤكدة حول هذا الموضوع. يرجى المحاولة مرة أخرى لاحقًا أو تقديم تفاصيل أكثر تحديدًا.",
        sources=[]
    )

# =============================================================================
# AI Components
# =============================================================================
@log_entry_exit
def moderate_query(query: str) -> bool:
    """Content moderation check"""
    try:
        response = app.state.openai_client.moderations.create(input=query)
        return not response.results[0].flagged
    except Exception as e:
        logger.error(f"Moderation error: {str(e)}")
        return True  # Fail-safe
@log_entry_exit
async def extract_keywords(query: str) -> List[str]:
    keyword_prompt = (
        "Analyze the following user query with a focus on LOCAL CONTEXT and CONCRETE ENTITIES.\n"
        "Extract 3 search terms (1-2 words each) following these rules:\n"
        "1. UNDERSTAND THE NATURE OF SOCIAL MEDIA PLATFORMS POSTS: Focus on capturing the topic of the query and provide keywords that people could use in their posts that answer the user’s query.\n"
        "2. PRIORITIZE SPECIFIC ENTITIES: Include exact names (brands/places/items…ect).\n"
        "3. COMBINE ENTITY + ACTION/EVENT: Like 'سوق حراء الشارع الرئيسي' or 'مهرجان الرياض للقهوة'.\n"
        "4. CAPTURE COLLOQUIAL PHRASES: Preserve local spellings (excluding typos).\n"
        "5. FILTER GENERIC TERMS: Reject vague words like 'events', always be hyper-specific.\n"
        "6. BALANCE SPECIFICITY WITH BREVITY: and account for informal language.\n"
        "- Must always include any city or location mentioned in the query.\n\n"
        "Examples:\n"
        "1- ايش اخر تعاونات ويستر؟"
        "Good Keywords:"
        "ويستر، تعاونات ويستر، تعاون ويستر، جديد ويستر، عرض ويستر"
        "2- هل دخول حديقة الوطن مجاني؟"
        "Good Keywords:"
        "حديقة الوطن، دخول حديقة الوطن، اسعار حديقة الوطن، سعر حديقة الوطن، حديقة الوطن مجاني"
        "3- ايش برامج تطوير الخريجين اللي تسجيلها مفتوح إلى الان؟"
        "Good Keywords:"
        "تطوير الخريجين، برامج تطوير الخريجين، برنامج تطوير الخريجين،  برامج تدريبية، برنامج تدريب"
        "Other examples:"
        "Query: أفضل كوفي في الخبر؟\n"
        "Good Keywords: أفضل كوفي الخبر, كوفي الخبر, مقاهي الخبر\n\n"
        "Query: أفضل مطعم بيتزا في الرياض\n"
        "Good Keywords: أفضل بيتزا الرياض, مطاعم بيتزا الرياض, بيتزا الرياض\n\n"
        f"Now your turn:\n"
        "Format: comma-separated terms, NO explanations.\n"
        f"Query: {query}\n"
    )
 
    response = await app.state.kw_model.ainvoke(keyword_prompt)

    keywords = clean_keywords(response.content)

    return keywords

def clean_keywords(keyword_string: str) -> List[str]:
    """Parse and clean keywords from model response"""
    # Normalize Arabic/English commas and split
    normalized = keyword_string.replace('،', ',')  # Convert Arabic commas to English
    cleaned = re.sub(r"[^\u0600-\u06FF, ]+", "", normalized)
    # Split and clean
    return [
        kw.strip()
        for kw in cleaned.split(",")
        if kw.strip() and len(kw.strip()) > 2
    ][:settings.max_keywords]

@log_entry_exit
def _update_keyword_data(current_data):
    """Transaction handler for incrementing or creating entries"""
    if current_data is None:
        # New keyword - create entry
        return {
            'volume': 1,
            'first_seen': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    else:
        # Existing keyword - increment count
        return {
            'volume': current_data.get('volume', 0) + 1,
            'last_updated': datetime.now().isoformat()
        }
@log_entry_exit
def generate_response(results: Dict, query: str) -> QueryResponse:
    """Generate final answer from results"""
    context = "\n".join(results["documents"][0])
    sources = list(set(
        meta["url"] for meta in results["metadatas"][0] if meta.get("url")
    ))

    system_rules = (
        "1. You are an AI assistant specialized in answering questions using social media posts written in the Saudi dialect\n"
        "2. Your primary goal is to facilitate access to local Saudi information\n"
        "3. Base answers ONLY on the provided social media content\n"
        "4. Preserve Saudi dialect terms and expressions\n"
        "5. Never add unverified information\n"
        "6. If no relevant posts exist, state 'No confirmed local information found'\n"
        "7. Format: Clear Arabic paragraphs with bullet points when listing\n"
    )

    final_prompt =(
        f"System Rules:{system_rules}\n"
        f"Follow these steps to answer the user query:\n"
        f"1. CONTEXT:\n{context}\n\n"
        f"2. QUERY: {query}\n\n"
        "RULES:\n"
        "a) Verify if query relates to Saudi local information\n"
        "b) Check context for EXACT matches to query entities\n"
        "c) If multiple conflicting posts exist, state 'No confirmed local information found'\n"
        "d) Compose answer using ONLY context from step 1\n"
        "e) Format: 1-3 concise paragraphs with Saudi dialect\n"
        "f) If no matches, return Rule 6 from system rules"
    )
    
    response = app.state.chat_model.invoke(final_prompt)
    answer = response.content
    
    return QueryResponse(
        answer=answer,
        sources=sources
    )

# =============================================================================
# Social Media Search Functions
# =============================================================================
@log_entry_exit
async def search_social_media(keywords: List[str],apify_client: ApifyClient) -> List[tuple]:
    """Async search of multiple platforms with batched keywords"""
    try:      
        # Create platform-specific search parameters
        twitter_search_terms = [f"{kw} since:2024-01-01_00:00:00_UTC" for kw in keywords]
        tiktok_search_queries = keywords  # Use raw keywords for TikTok

        # Run platform searches in parallel
        twitter_results, tiktok_results = await asyncio.gather(
            asyncio.to_thread(
                search_platform,
                platform='twitter',
                client=apify_client,
                search_terms=twitter_search_terms,
                max_results=settings.max_resultsx
            ),
            asyncio.to_thread(
                search_platform,
                platform='tiktok',
                client=apify_client,
                search_queries=tiktok_search_queries,
                max_results=settings.max_resultstiktok
            ),
            return_exceptions=True
        )

        # Process results
        results = []
        for platform, items in [('twitter', twitter_results), ('tiktok', tiktok_results)]:
            if isinstance(items, Exception):
                logger.error(f"{platform.capitalize()} search failed: {str(items)}")
            elif items:
                results.extend((platform, item) for item in items)
        
        return results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return []

@log_entry_exit
def search_platform(platform: str,client: ApifyClient, max_results: int,search_terms: Optional[List[str]] = None,search_queries: Optional[List[str]] = None) -> List[Dict]:
    """Search a single platform with batched keywords"""
    try:
        if platform == 'twitter' and search_terms:
            actor_call = client.actor("CJdippxWmn9uRfooo").call(run_input={
                "searchTerms": search_terms,
                "maxItems": max_results,
                "queryType": "Top",
                "lang": "ar",
            })
        elif platform == 'tiktok' and search_queries:
            actor_call = client.actor("OtzYfK1ndEGdwWFKQ").call(run_input={
                "searchQueries": search_queries,
                "resultsPerPage": max_results,
                "searchSection": "/video",
                "shouldDownloadVideos": False,
            })
        else:
            return []
            
        return list(client.dataset(actor_call["defaultDatasetId"]).iterate_items())
    except Exception as e:
        logger.error(f"{platform.capitalize()} search failed: {str(e)}")
        return []
    
@app.get("/")
def root():
    return {"status":"ok"}
    
'''if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)'''
