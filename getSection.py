import re
from difflib import SequenceMatcher

# Sample data
import json

# Open and read the JSON file
with open('act_sections.json', 'r') as file:
    data = json.load(file)

# Common words to ignore in matching
STOP_WORDS = {
    'the', 'of', 'and', 'in', 'to', 'for', 'a', 'an', 'on', 'at', 'by',
    'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'regulatory', 'authority'
}

# Minimum number of matching words required
MIN_MATCHING_WORDS = 1

def extract_section_and_act(query):
    # Match things like "Section 2 of the Registration Act" or "Section 2 Registration Act"
    pattern = r".*?[Ss]ection\s+(\d+[A-Za-z]*)\s+(?:of\s+)?(?:the\s+)?([^.?]+?)(?:\?|\.|$)"
    print("Query:", query)
    print("Pattern:", pattern)
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        section = match.group(1).strip()
        act = match.group(2).strip()
        # Remove any leading 'of' from the act name
        act = re.sub(r'^of\s+', '', act, flags=re.IGNORECASE)
        print("Extracted Section:", section)
        print("Extracted Act:", act)
        return section, act
    return None, None

def filter_stop_words(words):
    return [word.lower() for word in words if word.lower() not in STOP_WORDS]

def get_act_variations(act_name):
    """Get all variations of an act name including abbreviations."""
    variations = {
        'full_name': set(),
        'words': set(),
        'abbreviations': set()
    }
    
    # Add the original act name
    variations['full_name'].add(act_name.lower())
    
    # Add words from the act name
    words = act_name.split()
    variations['words'].update(word.lower() for word in words if word.lower() not in STOP_WORDS)
    
    # Add abbreviations if they exist in the data
    if isinstance(data[act_name], dict) and 'abbreviations' in data[act_name]:
        variations['abbreviations'].update(abbr.lower() for abbr in data[act_name]['abbreviations'])
    
    return variations

def calculate_match_score(query_words, act_name):
    # Convert both to lowercase and filter stop words
    query_words = set(filter_stop_words(query_words))
    act_variations = get_act_variations(act_name)
    
    # Skip if no meaningful words left after filtering
    if not query_words:
        return 0, set()
    
    # Calculate matches for each type
    full_name_matches = query_words.intersection(act_variations['full_name'])
    word_matches = query_words.intersection(act_variations['words'])
    abbreviation_matches = query_words.intersection(act_variations['abbreviations'])
    
    # Combine all matches for minimum word check
    all_matches = full_name_matches.union(word_matches).union(abbreviation_matches)
    
    # Return 0 score if we don't have enough matching words
    if len(all_matches) < MIN_MATCHING_WORDS:
        return 0, set()
    
    # Calculate weighted scores for each type of match
    # For abbreviations, we want to give a high score even if it's just one word
    if abbreviation_matches:
        abbreviation_score = 0.9  # High score for abbreviation match
    else:
        abbreviation_score = 0
    
    # For word matches, calculate based on how many words match
    word_score = len(word_matches) / len(query_words) if query_words else 0
    
    # For full name matches, calculate based on how much of the query matches
    full_name_score = len(full_name_matches) / len(query_words) if query_words else 0
    
    # Calculate non-matching words penalty
    act_words = set(filter_stop_words(act_name.split()))
    non_matching_words = act_words - query_words
    penalty_factor = 1.0
    if non_matching_words:
        # Penalty increases with the number of non-matching words
        # But is capped at 0.5 to not completely eliminate good matches
        penalty_factor = max(0.5, 1.0 - (len(non_matching_words) * 0.1))
    
    # Apply weights to different types of matches
    # Full name matches get highest weight (1.0)
    # Abbreviation matches get second highest weight (0.9)
    # Word matches get normal weight (0.7)
    weighted_score = (
        (full_name_score * 1.0) +
        (abbreviation_score * 0.9) +
        (word_score * 0.7)
    ) / 2.6  # Normalize by sum of weights
    
    # Apply penalty to weighted score
    weighted_score *= penalty_factor
    
    # Calculate string similarity score (using filtered words)
    similarity_score = SequenceMatcher(None, ' '.join(query_words), ' '.join(act_variations['words'])).ratio()
    
    # Combine scores (70% weight to word matches, 30% to string similarity)
    final_score = (weighted_score * 0.7) + (similarity_score * 0.3)
    
    return final_score, all_matches

def find_section_title(query, data):
    section, act = extract_section_and_act(query)
    if section and act:
        # Split the act name into words
        act_words = act.split()
        filtered_words = filter_stop_words(act_words)
        print("\nAct words from query (after filtering):", filtered_words)
        
        # Find all matching acts with scores
        matching_acts = []
        for act_name in data:
            score, matched_words = calculate_match_score(act_words, act_name)
            if score > 0.3 and len(matched_words) >= MIN_MATCHING_WORDS:  # Check both score and minimum words
                matching_acts.append({
                    "act_name": act_name,
                    "score": score,
                    "matched_words": matched_words
                })
        
        # Sort matching acts by score
        matching_acts.sort(key=lambda x: x["score"], reverse=True)
        print("\nMatching acts found (sorted by score):")
        for match in matching_acts:
            print(f"Act: {match['act_name']}, Score: {match['score']:.2f}, Matched words: {match['matched_words']}")
        
        # If we found matching acts, search for the section in all of them
        if matching_acts:
            section_titles = []
            for match in matching_acts:
                act_name = match["act_name"]
                # Handle both old and new data structure
                sections = data[act_name].get('sections', data[act_name]) if isinstance(data[act_name], dict) else data[act_name]
                for entry in sections:
                    if entry["section"].lower() == section.lower():
                        section_titles.append({
                            "act": act_name,
                            "section": section,
                            "title": entry["section_title"],
                            "match_score": match["score"],
                            "matched_words": match["matched_words"]
                        })
            
            if section_titles:
                return section_titles
            return f"No matching section {section} found in any of the matching acts"
        return f"No matching acts found for: {act}"
    return "No matching section found."

def getSectionTitle(query):
    result = find_section_title(query, data)
    return result

# result = getSectionTitle(input("Enter query:"))
# print("\nResults:")
# for item in result if isinstance(result, list) else [result]:
#     print(item)