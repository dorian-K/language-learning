import json
import os
from collections import Counter

# INPUT = os.path.join(os.path.dirname(__file__), "../anki/Refold ES1K/")
INPUT = os.path.join(os.path.dirname(__file__), "../vocab/lt")


def calculate_level_statistics(directory_path):
    # Initialize counters for the levels
    earliest_counts = Counter()
    mandatory_counts = Counter()
    total_files = 0
    total_cards = 0

    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # Iterate over all JSON files in the given directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)

            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)

                    total_files += 1

                    # Based on the schema, data is a list of objects
                    if isinstance(data, list):
                        for card in data:
                            total_cards += 1

                            earliest = card.get("earliest_level")
                            mandatory = card.get("mandatory_level")

                            # Increment counts if the levels are present
                            if earliest:
                                earliest_counts[earliest] += 1
                            if mandatory:
                                mandatory_counts[mandatory] += 1
                                # if mandatory == "C1":
                                #    print(json.dumps(card, indent=2, ensure_ascii=False))
                    else:
                        print(f"Warning: {filename} does not contain a JSON array.")

            except json.JSONDecodeError:
                print(f"Error reading {filename}: Invalid JSON format. Skipping.")
            except Exception as e:
                print(f"Error reading {filename}: {e}. Skipping.")

    # Standard CEFR levels for sorting the output
    cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

    # Print the statistics
    print("========================================")
    print("          FLASHCARD STATISTICS          ")
    print("========================================")
    print(f"Total word pairs (files): {total_files}")
    print(f"Total individual cards:   {total_cards}")
    print("========================================\n")

    print("Distribution by 'Earliest Level':")
    print("---------------------------------")
    for level in cefr_levels:
        count = earliest_counts.get(level, 0)
        print(f"  {level}: {count} cards")

    print("\nDistribution by 'Mandatory Level':")
    print("----------------------------------")
    for level in cefr_levels:
        count = mandatory_counts.get(level, 0)
        print(f"  {level}: {count} cards")
    print("\n========================================")

    # any levels that are not in the standard CEFR list will be printed at the end
    other_earliest_levels = {k: v for k, v in earliest_counts.items() if k not in cefr_levels}
    other_mandatory_levels = {k: v for k, v in mandatory_counts.items() if k not in cefr_levels}
    if other_earliest_levels:
        print("\nOther 'Earliest Levels' found:")
        for level, count in other_earliest_levels.items():
            print(f"  {level}: {count} cards")
    if other_mandatory_levels:
        print("\nOther 'Mandatory Levels' found:")
        for level, count in other_mandatory_levels.items():
            print(f"  {level}: {count} cards")


if __name__ == "__main__":
    calculate_level_statistics(INPUT)
