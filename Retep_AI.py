import streamlit as st
import urllib.request
import urllib.error
import json
import chromadb
import uuid
import hashlib

from groq import Groq
from pypdf import PdfReader
from io import BytesIO


# API keys

def get_secret(name):
    try:
        return st.secrets[name]
    except KeyError:
        return None


GEMINI_API_KEY = get_secret('GEMINI_API_KEY')
GROQ_API_KEY = get_secret('GROQ_API_KEY')


# Models

MODELS = {}

if GEMINI_API_KEY:
    MODELS.update({
        'Gemini 3.6 Flash': ('gemini', 'gemini-3.6-flash'),
        'Gemini 3.5 Flash': ('gemini', 'gemini-3.5-flash'),
        'Gemini 3.5 Flash-Lite': ('gemini', 'gemini-3.5-flash-lite')
    })

if GROQ_API_KEY:
    MODELS.update({
        'Groq GPT-OSS 120B': ('groq', 'openai/gpt-oss-120b'),
        'Groq GPT-OSS 20B': ('groq', 'openai/gpt-oss-20b'),
        'Groq Qwen 3.6 27B': ('groq', 'qwen/qwen3.6-27b')
    })

if not MODELS:
    st.error(
        'No API keys found. Add GEMINI_API_KEY or '
        'GROQ_API_KEY to Streamlit secrets.'
    )
    st.stop()


# Page

st.title('Retep AI')

selected_model = st.sidebar.selectbox(
    'Model',
    list(MODELS.keys())
)

PROVIDER, MODEL = MODELS[selected_model]

st.caption(f'Current model: {selected_model}')


# Groq client

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )
else:
    groq_client = None


# Session state

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = (
        'docs_' + uuid.uuid4().hex[:12]
    )

if 'collection' not in st.session_state:
    st.session_state.collection = (
        st.session_state.chroma_client.create_collection(
            name=st.session_state.collection_name
        )
    )

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}


# File uploader

files = st.file_uploader(
    'Upload files',
    type=None,
    accept_multiple_files=True,
    key='knowledge_upload_v2'
)


# Read files

def read_file(file_bytes, filename):
    extension = (
        filename.rsplit('.', 1)[-1].lower()
        if '.' in filename
        else ''
    )

    if extension == 'pdf':
        reader = PdfReader(
            BytesIO(file_bytes)
        )

        pages = []

        for page in reader.pages:
            page_text = (
                page.extract_text()
                or ''
            )

            if page_text.strip():
                pages.append(
                    page_text
                )

        return '\n\n'.join(
            pages
        )

    try:
        return file_bytes.decode(
            'utf-8'
        )

    except UnicodeDecodeError:
        try:
            return file_bytes.decode(
                'cp1252'
            )

        except UnicodeDecodeError:
            return file_bytes.decode(
                'utf-8',
                errors='replace'
            )


# Process files

if files and st.button('Process Files'):
    collection = st.session_state.collection
    total_new_chunks = 0
    added_files = []
    skipped_files = []

    with st.spinner('Processing files...'):
        for file in files:
            try:
                file_bytes = file.getvalue()

                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                if file_hash in st.session_state.processed_files:
                    skipped_files.append(
                        file.name
                    )
                    continue

                text = read_file(
                    file_bytes,
                    file.name
                )

                if not text.strip():
                    st.warning(
                        f'No readable text found in {file.name}.'
                    )
                    continue

                chunks = []
                chunk_size = 300
                overlap = 200
                step = chunk_size - overlap

                for i in range(
                    0,
                    len(text),
                    step
                ):
                    chunk = text[
                        i:i + chunk_size
                    ].strip()

                    if chunk:
                        chunks.append(
                            chunk
                        )

                ids = [
                    f'{file_hash}_{i}'
                    for i in range(
                        len(chunks)
                    )
                ]

                metadatas = [
                    {
                        'filename': file.name,
                        'file_hash': file_hash,
                        'chunk': i
                    }
                    for i in range(
                        len(chunks)
                    )
                ]

                collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )

                st.session_state.processed_files[
                    file_hash
                ] = file.name

                total_new_chunks += len(
                    chunks
                )

                added_files.append(
                    file.name
                )

            except Exception as e:
                st.error(
                    f'Could not process {file.name}: {e}'
                )

    if added_files:
        st.success(
            f'Added {len(added_files)} file(s) and '
            f'{total_new_chunks} chunks to the knowledge base.'
        )

    if skipped_files:
        st.info(
            'Already in knowledge base: '
            + ', '.join(skipped_files)
        )


# Knowledge base status

if st.session_state.processed_files:
    file_names = list(
        st.session_state.processed_files.values()
    )

    st.caption(
        f'Knowledge base: {len(file_names)} file(s)'
    )

    with st.expander(
        'Files in knowledge base'
    ):
        for name in file_names:
            st.write(name)


# Build current prompt

def build_current_prompt(
    context,
    question
):
    if not context:
        return question

    return f'''
Relevant information retrieved from the user's knowledge base:

{context}

User message:

{question}

Use the retrieved information when it is relevant.
You may answer normally using your general knowledge when the retrieved information is not relevant.
'''


# Gemini messages

def gemini_messages(
    context,
    question
):
    contents = []

    for message in st.session_state.messages[:-1]:
        role = (
            'model'
            if message['role'] == 'assistant'
            else 'user'
        )

        contents.append({
            'role': role,
            'parts': [
                {
                    'text': message['content']
                }
            ]
        })

    contents.append({
        'role': 'user',
        'parts': [
            {
                'text': build_current_prompt(
                    context,
                    question
                )
            }
        ]
    })

    return contents


# Groq messages

def groq_messages(
    context,
    question
):
    messages = [
        {
            'role': 'system',
            'content': (
                'You are Retep AI. '
                'Answer normally and conversationally. '
                'You may receive information retrieved '
                'from the user knowledge base. '
                'Use that information when relevant. '
                'Do not pretend retrieved information '
                'contains something that it does not.'
            )
        }
    ]

    for message in st.session_state.messages[:-1]:
        messages.append({
            'role': message['role'],
            'content': message['content']
        })

    messages.append({
        'role': 'user',
        'content': build_current_prompt(
            context,
            question
        )
    })

    return messages


# Gemini response

def gemini_response(
    context,
    question
):
    api_url = (
        f'https://generativelanguage.googleapis.com/v1beta/'
        f'models/{MODEL}:streamGenerateContent?alt=sse'
    )

    body = {
        'systemInstruction': {
            'parts': [
                {
                    'text': (
                        'You are Retep AI. '
                        'Answer normally and conversationally. '
                        'You may receive information retrieved '
                        'from the user knowledge base. '
                        'Use that information when relevant. '
                        'Do not pretend retrieved information '
                        'contains something that it does not.'
                    )
                }
            ]
        },
        'contents': gemini_messages(
            context,
            question
        )
    }

    request = urllib.request.Request(
        api_url,
        data=json.dumps(
            body
        ).encode('utf-8'),
        headers={
            'x-goog-api-key': GEMINI_API_KEY,
            'Content-Type': 'application/json'
        },
        method='POST'
    )

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:

        for raw_line in response:
            line = raw_line.decode(
                'utf-8'
            ).strip()

            if not line:
                continue

            if not line.startswith(
                'data:'
            ):
                continue

            data = line[5:].strip()

            try:
                chunk = json.loads(
                    data
                )

            except json.JSONDecodeError:
                continue

            candidates = chunk.get(
                'candidates',
                []
            )

            if not candidates:
                continue

            content = candidates[0].get(
                'content',
                {}
            )

            for part in content.get(
                'parts',
                []
            ):
                text = part.get(
                    'text'
                )

                if text:
                    yield text


# Groq response

def groq_response(
    context,
    question
):
    stream = (
        groq_client
        .chat
        .completions
        .create(
            model=MODEL,
            messages=groq_messages(
                context,
                question
            ),
            stream=True
        )
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        text = (
            chunk
            .choices[0]
            .delta
            .content
        )

        if text:
            yield text


# Response generator

def response_generator(
    context=None,
    question=None
):
    try:
        if PROVIDER == 'gemini':
            yield from gemini_response(
                context,
                question
            )

        elif PROVIDER == 'groq':
            yield from groq_response(
                context,
                question
            )

    except urllib.error.HTTPError as e:
        error_body = e.read().decode(
            'utf-8',
            errors='replace'
        )

        try:
            error_data = json.loads(
                error_body
            )

            error = error_data.get(
                'error',
                {}
            )

            if isinstance(
                error,
                dict
            ):
                error_message = error.get(
                    'message',
                    error_body
                )
            else:
                error_message = str(
                    error
                )

        except Exception:
            error_message = (
                error_body
                if error_body
                else f'HTTP error {e.code}'
            )

        yield (
            f'\n\nAPI Error {e.code}: '
            f'{error_message}'
        )

    except Exception as e:
        status_code = getattr(
            e,
            'status_code',
            None
        )

        body = getattr(
            e,
            'body',
            None
        )

        if status_code:
            if body:
                yield (
                    f'\n\nAPI Error {status_code}: '
                    f'{body}'
                )
            else:
                yield (
                    f'\n\nAPI Error {status_code}: '
                    f'{str(e)}'
                )
        else:
            yield (
                '\n\nAPI Error: '
                f'{str(e)}'
            )


# Display chat history

for message in st.session_state.messages:
    with st.chat_message(
        message['role']
    ):
        st.markdown(
            message['content']
        )


# Chat

if prompt := st.chat_input(
    'Ask me something...'
):
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

    with st.chat_message(
        'user'
    ):
        st.markdown(
            prompt
        )

    context = None
    collection = st.session_state.collection
    number_of_chunks = collection.count()

    if number_of_chunks > 0:
        try:
            n_results = min(
                5,
                number_of_chunks
            )

            result = collection.query(
                query_texts=[
                    prompt
                ],
                n_results=n_results
            )

            retrieved_chunks = (
                result['documents'][0]
            )

            retrieved_metadata = (
                result['metadatas'][0]
            )

            context_parts = []

            for chunk, metadata in zip(
                retrieved_chunks,
                retrieved_metadata
            ):
                filename = metadata.get(
                    'filename',
                    'Unknown file'
                )

                context_parts.append(
                    f'Source: {filename}\n'
                    f'{chunk}'
                )

            context = '\n\n'.join(
                context_parts
            )

        except Exception as e:
            st.warning(
                'Could not search the '
                f'knowledge base: {e}'
            )

    with st.chat_message(
        'assistant'
    ):
        response = st.write_stream(
            response_generator(
                context=context,
                question=prompt
            )
        )

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })
