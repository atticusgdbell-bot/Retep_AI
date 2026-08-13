import streamlit as st
import urllib.request
import urllib.error
import json
import chromadb
import uuid
import hashlib

from pypdf import PdfReader
from io import BytesIO


# Config

MODEL = 'gemini-3.5-flash'

API_URL = (
    f'https://generativelanguage.googleapis.com/v1beta/'
    f'models/{MODEL}:streamGenerateContent?alt=sse'
)

try:
    API_KEY = st.secrets['GEMINI_API_KEY']
except KeyError:
    st.error('GEMINI_API_KEY is missing.')
    st.stop()


# Page

st.title('Retep AI')
st.caption(f'Current model: {MODEL}')


# Session state

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = 'docs_' + uuid.uuid4().hex[:12]

if 'collection' not in st.session_state:
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name=st.session_state.collection_name
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
# Read a file

def read_file(file_bytes, filename):
    extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if extension == 'pdf':
        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ''

            if page_text.strip():
                pages.append(page_text)

        return '\n\n'.join(pages)

    # Most modern text files
    try:
        return file_bytes.decode('utf-8')

    # Some older Windows text files
    except UnicodeDecodeError:
        try:
            return file_bytes.decode('cp1252')
        except UnicodeDecodeError:
            return file_bytes.decode('utf-8', errors='replace')


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
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                if file_hash in st.session_state.processed_files:
                    skipped_files.append(file.name)
                    continue

                text = read_file(file_bytes, file.name)

                if not text.strip():
                    st.warning(f'No readable text found in {file.name}.')
                    continue

                chunks = []
                chunk_size = 300
                overlap = 200
                step = chunk_size - overlap

                for i in range(0, len(text), step):
                    chunk = text[i:i + chunk_size].strip()

                    if chunk:
                        chunks.append(chunk)

                ids = [
                    f'{file_hash}_{i}'
                    for i in range(len(chunks))
                ]

                metadatas = [
                    {
                        'filename': file.name,
                        'file_hash': file_hash,
                        'chunk': i
                    }
                    for i in range(len(chunks))
                ]

                collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )

                st.session_state.processed_files[file_hash] = file.name
                total_new_chunks += len(chunks)
                added_files.append(file.name)

            except Exception as e:
                st.error(f'Could not process {file.name}: {e}')

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
    file_names = list(st.session_state.processed_files.values())

    st.caption(
        f'Knowledge base: {len(file_names)} file(s)'
    )

    with st.expander('Files in knowledge base'):
        for name in file_names:
            st.write(name)


# Convert chat history to Gemini format

def gemini_messages(context=None, question=None):
    contents = []

    for message in st.session_state.messages[:-1]:
        role = 'model' if message['role'] == 'assistant' else 'user'

        contents.append(
            {
                'role': role,
                'parts': [
                    {
                        'text': message['content']
                    }
                ]
            }
        )

    if context:
        current_prompt = f'''
Relevant information retrieved from the user's knowledge base:

{context}

User message:

{question}

Use the retrieved information when it is relevant.
You may answer normally using your general knowledge when the retrieved information is not relevant.
'''
    else:
        current_prompt = question

    contents.append(
        {
            'role': 'user',
            'parts': [
                {
                    'text': current_prompt
                }
            ]
        }
    )

    return contents


# Gemini streaming response

def response_generator(context=None, question=None):
    body = {
        'systemInstruction': {
            'parts': [
                {
                    'text': (
                        'You are Retep AI. '
                        'Answer normally and conversationally. '
                        'You may receive information retrieved from '
                        'the user knowledge base. '
                        'Use that information when relevant. '
                        'Do not pretend retrieved information contains '
                        'something that it does not.'
                    )
                }
            ]
        },
        'contents': gemini_messages(context, question)
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'x-goog-api-key': API_KEY,
            'Content-Type': 'application/json'
        },
        method='POST'
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode('utf-8').strip()

                if not line:
                    continue

                if not line.startswith('data:'):
                    continue

                data = line[5:].strip()

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                candidates = chunk.get('candidates', [])

                if not candidates:
                    continue

                content = candidates[0].get('content', {})

                for part in content.get('parts', []):
                    text = part.get('text')

                    if text:
                        yield text

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode('utf-8')
            error_data = json.loads(error_body)

            error_message = (
                error_data
                .get('error', {})
                .get('message', 'Unknown API error')
            )

        except Exception:
            error_message = f'HTTP error {e.code}'

        yield f'\n\nAPI Error {e.code}: {error_message}'

    except urllib.error.URLError as e:
        yield f'\n\nConnection error: {e.reason}'

    except TimeoutError:
        yield '\n\nRequest timed out. Please try again.'

    except Exception as e:
        yield f'\n\nUnexpected error: {str(e)}'


# Display chat history

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# Chat

if prompt := st.chat_input('Ask me something...'):
    st.session_state.messages.append(
        {
            'role': 'user',
            'content': prompt
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    context = None
    collection = st.session_state.collection
    number_of_chunks = collection.count()

    if number_of_chunks > 0:
        try:
            n_results = min(5, number_of_chunks)

            result = collection.query(
                query_texts=[prompt],
                n_results=n_results
            )

            retrieved_chunks = result['documents'][0]
            retrieved_metadata = result['metadatas'][0]

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
                    f'Source: {filename}\n{chunk}'
                )

            context = '\n\n'.join(context_parts)

        except Exception as e:
            st.warning(
                f'Could not search the knowledge base: {e}'
            )

    with st.chat_message('assistant'):
        response = st.write_stream(
            response_generator(
                context=context,
                question=prompt
            )
        )

    st.session_state.messages.append(
        {
            'role': 'assistant',
            'content': response
        }
    )
