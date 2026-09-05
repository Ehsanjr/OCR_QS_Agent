import os

# ⚠️ PLACEHOLDER OPTIONS
# The real sample question blocks (with actual multiple-choice options and
# ground-truth answers) referenced in the take-home brief were not included
# among the files reviewed, so these are stand-ins for local pipeline
# testing only. Replace with the real options for each block before
# generating your submission's output.
MOCK_OPTIONS_BY_BLOCK = {
    "q113.png": {
        "A": "گزینه ۱ (این مقدار را با گزینه واقعی جایگزین کنید)",
        "B": "گزینه ۲ (این مقدار را با گزینه واقعی جایگزین کنید)",
        "C": "گزینه ۳ (این مقدار را با گزینه واقعی جایگزین کنید)",
        "D": "گزینه ۴ (این مقدار را با گزینه واقعی جایگزین کنید)",
    }
}

DEFAULT_MOCK_OPTIONS = {
    "A": "گزینه A",
    "B": "گزینه B",
    "C": "گزینه C",
    "D": "گزینه D",
}


def get_cached_ocr_text(image_path: str) -> str:
    """
    Mock function to return a hardcoded OCR result and save API credits during testing.
    Reads from a text file if it exists, otherwise creates it with the hardcoded text.
    """
    cache_file = image_path + ".txt"

    # 1. If cache file exists, read and return its content
    if os.path.exists(cache_file):
        print(f"[Info] Loading OCR result from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # 2. Hardcoded text for q113.png
    hardcoded_text = """۱۱۳- تابع f(x)=mx -nx-k در هر بازه، هم صعودی و هم نزولی است. اگر مجموعه زیر، تابع باشد. مقدار
کدام است<math>f(\sqrt{\Delta})</math>
<math>\{(m,n-1),(\circ,k),(n-1,m^{\dagger}+tm-1),(t^{\dagger}k+t,t^{\dagger}k+1)\}</math>
VA (F
(1
<math>=1.0</math>"""

    # 3. Write to cache file for future uses
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(hardcoded_text)
        print(f"[Info] Saved mock OCR text to cache: {cache_file}")

    return hardcoded_text


def get_mock_options(image_path: str) -> dict:
    """
    Returns placeholder multiple-choice options for local testing.
    ⚠️ Replace with the real options for each block before evaluating.
    """
    key = os.path.basename(image_path)
    return MOCK_OPTIONS_BY_BLOCK.get(key, dict(DEFAULT_MOCK_OPTIONS))