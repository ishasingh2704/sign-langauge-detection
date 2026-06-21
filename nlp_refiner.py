from openai import OpenAI
import time
import requests

LM_STUDIO_BASE = "http://localhost:1234/v1"

client = OpenAI(
    base_url=LM_STUDIO_BASE,
    api_key="lm-studio"
)

def test_lm_studio_connection():
    """Test if LM Studio server is accessible"""
    try:
        response = requests.get(f"{LM_STUDIO_BASE}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return True, f"Connected! Available models: {len(models.get('data', []))}"
        else:
            return False, f"Server responded with status {response.status_code}"

    # ✅ FIXED exception
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - LM Studio not running"

    except requests.exceptions.Timeout:
        return False, "Connection timeout - server too slow"

    except Exception as e:
        return False, f"Connection error: {str(e)}"


def preprocess(chars):
    vowels = set("aeiou")
    result = []
    current = ""

    for c in chars:
        if len(current) >= 2 and all(ch not in vowels for ch in current[-2:]) and c not in vowels:
            result.append(current)
            current = c
        else:
            current += c

    if current:
        result.append(current)

    return " ".join(result)


def refine_buffer(buffer):
    return refine_asl_buffer(buffer)


def refine_asl_buffer(buffer_input):
    start_time = time.time()

    if isinstance(buffer_input, str):
        buffer = buffer_input.split() if ' ' in buffer_input else list(buffer_input)
    else:
        buffer = buffer_input

    cleaned_buffer = [char.lower() for char in buffer if char and char.strip()]
    cleaned = ''.join(cleaned_buffer)

    preprocessed = preprocess(cleaned_buffer)

    connected, connection_msg = test_lm_studio_connection()
    if not connected:
        raise Exception(f"LM Studio connection failed: {connection_msg}")

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b-instruct-1m",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ASL buffer decoder.\n"
                        "- Add spaces to form valid English words.\n"
                        "- Do NOT hallucinate.\n"
                        "- Remove unnecessary repeated letters.\n"
                        "- Output only the corrected English sentence."
                    )
                },
                {
                    "role": "user",
                    "content": f"Decode this ASL letter sequence into proper English: {preprocessed}"
                }
            ],
            temperature=0.1
        )

        refined_text = response.choices[0].message.content.strip()

    except Exception as e:
        raise Exception(f"LM Studio API error: {str(e)}")

    device = "lm-studio"
    processing_time = time.time() - start_time

    return {
        'refined_text': refined_text,
        'preprocessed': preprocessed,
        'cleaned': cleaned,
        'processing_time_seconds': round(processing_time, 3),
        'model_device': device,
        'connection_status': connection_msg
    }



