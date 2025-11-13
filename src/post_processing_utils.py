import re
from number_parser import parse

def convert_timestamps_to_numbers(text):
    """
    Converts timestamps in the format HH:MM or MM:SS into numbers by removing the colon.
    Example: "The time is 02:22" -> "The time is 222"
    """
    # Regular expression to match timestamps (e.g., 02:22, 12:34)
    timestamp_pattern = r'\b(\d{1,2}):(\d{2})\b'

    # Replace the colon in timestamps with an empty string and remove leading zeros
    processed_text = re.sub(
        timestamp_pattern,
        lambda m: str(int(m.group(1))) + str(int(m.group(2))),  # Convert the first group to int to remove leading zeros
        text
    )
    return processed_text


def process_text_with_mixed_numbers(text):
    """
    Processes text to handle cases like "nine 90" and converts them to "990".
    Handles punctuation and multiple sentences.
    """
    word_to_number = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9
}
    
    # Split the text into sentences based on punctuation
    sentences = re.split(r'([.!?])', text)  # Keep punctuation as separate tokens
    processed_sentences = []

    for i in range(0, len(sentences), 2):  # Process sentences and skip punctuation
        sentence = sentences[i].strip()
        if not sentence:
            continue

        words = sentence.split()
        result = []
        j = 0

        while j < len(words):
            word = words[j]
            normalized_word = word.lower()
            # Check if the current word is a single-digit number in words
            if normalized_word in word_to_number:
                # Check if the next word is a numeric digit (e.g., "90")
                if j + 1 < len(words) and words[j + 1].isdigit():
                    # Concatenate the current number with the next numeric value
                    combined_number = str(word_to_number[normalized_word]) + words[j + 1]
                    result.append(combined_number)
                    j += 2  # Skip the next word since it's already processed
                    continue

            # If no match, just add the word as-is
            result.append(word)
            j += 1

        processed_sentences.append(" ".join(result))

    # Reconstruct the text with punctuation
    reconstructed_text = ""
    for i in range(len(sentences)):
        if i % 2 == 0:  # Sentence part
            reconstructed_text += processed_sentences.pop(0) if processed_sentences else ""
        else:  # Punctuation part
            reconstructed_text += sentences[i]

    return reconstructed_text

def process_repeated_numbers(text):
    """
    Processes text to handle cases like "Triple 1 66 9 77" and "triple 123456".
    Converts repeated patterns like "triple 1" to "111" and "triple 123456" to "11123456".
    Handles case-insensitivity for words like "Triple", "One", etc.
    """
    # Mapping for single-digit numbers in words
    word_to_number = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }

    # Mapping for repeated patterns
    repeat_map = {
        "double": 2,  # Repeat the digit twice
        "triple": 3   # Repeat the digit three times
    }

    # Regex to match repeated patterns (e.g., "triple one", "double six", "triple 123456")
    pattern = r"(?i)(double|triple) (\d+|\w+)"  # (?i) makes the regex case-insensitive

    def replace_match(match):
        """
        Replaces matched patterns with their numeric equivalents.
        """
        repeat_word = match.group(1).lower()  # Normalize "Double" or "Triple" to lowercase
        number_word = match.group(2)  # Keep the original case for digits or words

        if number_word.isdigit():
            # If it's a digit or number, repeat only the first digit
            return number_word[0] * repeat_map[repeat_word] + number_word[1:]
        elif number_word.lower() in word_to_number:
            # If it's in words, convert to digit and repeat
            return word_to_number[number_word.lower()] * repeat_map[repeat_word]
        return match.group(0)  # Return the original text if no match

    # Process the text
    processed_text = re.sub(pattern, replace_match, text)

    return processed_text

def post_process_pharmacord(text):
    # Replace various accredo variants with "Accredo" (preserving trailing punctuation)
    accredo_variants = (
        r'(?:'
        r'A\s*C\s*C\s*R\s*E\s*D\s*O|'  # A C C R E D O (with any spacing)
        r'A\s*credo|'                 # A credo
        r'A\s*cradle|'                # A cradle
        r'a\s*Crea|'                  # a Crea
        r'O\s*Crito|'                 # O Crito
        r'a\s*credo\.?|'              # a credo with optional dot
        r'The\s+credo|'               # The credo
        r'Acrid|'                     # Acrid
        r'A\s*crater|'                # A crater
        r'Laredo|'                   # Laredo
        r'Accreta'                   # Accreta
        r')'
    )
    accredo_pattern = re.compile(r'\b' + accredo_variants + r'\b([.,!?:;]*)', re.IGNORECASE)
    text = accredo_pattern.sub(lambda m: "Accredo" + m.group(1), text)

    # Matches "A Caria" (with optional extra spaces) or "Aquaria" and converts to "Acaria"
    acaria_variants = r'(?:A\s*Caria|Aquaria)'
    acaria_pattern = re.compile(r'\b' + acaria_variants + r'\b([.,!?:;]*)', re.IGNORECASE)
    text = acaria_pattern.sub(lambda m: "Acaria" + m.group(1), text)

    # Replace "Centerville" with "CenterWell" preserving trailing punctuation.
    centerville_pattern = re.compile(r'\bCenterville\b([.,!?:;]*)', re.IGNORECASE)
    text = centerville_pattern.sub(lambda m: "CenterWell" + m.group(1), text)
    
    
    # Remove stray "O" after acronyms HMO/PPO (e.g., "HMO O." -> "HMO.")
    pattern_after = re.compile(r'\b(HMO|PPO)\s+O\b([.,!?]?)', re.IGNORECASE)
    text = pattern_after.sub(r'\1\2', text)

    # Remove stray "O" before acronyms HMO/PPO (e.g., "O PPO." -> "PPO.")
    pattern_before = re.compile(r'\bO\s+(HMO|PPO)\b([.,!?]?)', re.IGNORECASE)
    text = pattern_before.sub(r'\1\2', text)
    
    # Remove stray "S" after "POS" (e.g., "POS S." -> "POS.")
    pos_pattern_after = re.compile(r'\b(POS)\s+S\b([.,!?]?)', re.IGNORECASE)
    text = pos_pattern_after.sub(r'\1\2', text)
    
    # Remove stray "S" before "POS" (e.g., "S POS." -> "POS.")
    pos_pattern_before = re.compile(r'\bS\s+(POS)\b([.,!?]?)', re.IGNORECASE)
    text = pos_pattern_before.sub(r'\1\2', text)

    # Replace repeated tokens for PA or FE (separated by spaces or periods) with a single occurrence
    token_pattern = re.compile(r'\b((?P<token>PA|FE|PPO|POS|HMO|Accredo)(?:[ .]+(?P=token))+\b)', re.IGNORECASE)
    text = token_pattern.sub(lambda m: m.group("token"), text)
    
    return text

def post_process_itn_output(text):
    result = post_process_pharmacord(parse(process_text_with_mixed_numbers(process_repeated_numbers(convert_timestamps_to_numbers(str(text))))))
    return result


# text = "apartment eight O PPO Centerville a credo. PA PA PA is required. Acrid, and PA.PA.PA is needed. FE FE works too! Also, HMO O, O PPO, and PPO O. Additionally, POS S and S POS POS.POS are examples."
# post_process_itn_output(text)
    
