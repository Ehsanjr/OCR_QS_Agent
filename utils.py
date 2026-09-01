import random

def inject_synthetic_noise(text: str, num_changes: int = 3) -> str:
    """
    Replaces a few characters with visually similar Persian letters and numbers 
    to simulate OCR scanning errors.
    """
    if not text:
        return text

    # Dictionary of visually similar characters to simulate scanning errors
    confusion_map = {
        '۲': '۳',
        '۳': '۲',
        'پ': 'ب',
        'ب': 'پ',
        'صفر': 'صقر',
        'ز': 'ر',
        'ر': 'ز',
        'ا': '۱',
        '۱': 'ا'
    }
    
    text_list = list(text)
    changes_made = 0
    
    # Find all indices that contain characters present in the confusion_map
    possible_indices = [i for i, char in enumerate(text_list) if char in confusion_map]
    
    # If the number of modifiable characters is less than the requested limit, adjust the limit
    num_changes = min(num_changes, len(possible_indices))
    
    if num_changes > 0:
        # Randomly select positions to inject noise
        selected_indices = random.sample(possible_indices, num_changes)
        
        for idx in selected_indices:
            original_char = text_list[idx]
            text_list[idx] = confusion_map[original_char]
            changes_made += 1
            
    print(f"[Debug] Synthetic noise injected: {changes_made} characters altered.")
    return "".join(text_list)