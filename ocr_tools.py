import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")
API_URL = "https://www.datalab.to/api/v1/ocr"


def extract_text_with_datalab(image_path: str, max_polls: int = 60, poll_interval: float = 2.0) -> str:
    """
    Sends the image to Datalab OCR and returns the extracted text.
    """
    headers = {"X-Api-Key": DATALAB_API_KEY}

    try:
        # 1) Send OCR request
        with open(image_path, "rb") as image_file:
            form_data = {"file": (os.path.basename(image_path), image_file, "image/png")}
            response = requests.post(API_URL, files=form_data, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        

        if not data.get("success"):
            print(f"OCR submission failed: {data.get('error')}")
            return ""

        check_url = data["request_check_url"]

        # 2) Polling until processing is complete
        for _ in range(max_polls):
            poll_response = requests.get(check_url, headers=headers, timeout=30)
            poll_response.raise_for_status()
            result = poll_response.json()

            if result.get("status") == "complete":
                if not result.get("success"):
                    print(f"OCR failed: {result.get('error')}")
                    return ""
                return _extract_text_from_pages(result.get("pages", []))

            time.sleep(poll_interval)

        print("OCR timed out while polling for result.")
        return ""

    except Exception as e:
        print(f"Error in OCR API: {e}")
        return ""


def _extract_text_from_pages(pages: list) -> str:
    """
    Extracts line text from the pages structure of the Datalab response.
    """
    lines_text = [] 
    for page in pages:
        for line in page.get("text_lines", []):
            if line.get("text"):
                lines_text.append(line["text"])
    return "\n".join(lines_text)



if __name__ == "__main__":
    # Assuming you placed one of the 4 images in the data folder
    # Replace with your exact image file name here
    test_image_path = "data/q113.png" 
    
    if os.path.exists(test_image_path):
        print(f"Sending '{test_image_path}' to Datalab OCR...")
        print("Waiting for processing (Polling)...")
        
        start_time = time.time()
        result_text = extract_text_with_datalab(test_image_path)
        end_time = time.time()
        
        print("\n" + "="*40)
        print("✅ Text extraction completed successfully:")
        print("="*40)
        print(result_text)
        print("="*40)
        print(f"⏱ Processing time: {end_time - start_time:.2f} seconds")
        
    else:
        print(f"❌ Error: Image not found at '{test_image_path}'! Please check the path and filename.")