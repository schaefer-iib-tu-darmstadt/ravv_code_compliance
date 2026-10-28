"""Chat message builders (system prompt, document reference, base64 JPEG image)."""
import base64

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_messages_with_reference_and_image(system_prompt: str, question: str, reference: str, image_path: str) -> list[dict]:
    base64_image = encode_image(image_path)

    user_prompt_parts = [
        f"--- START OF DOCUMENT ---\n{reference}\n--- END OF DOCUMENT ---",
        f"Please answer the following question based on the document and the image: {question}"
    ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "\n\n".join(user_prompt_parts)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]

