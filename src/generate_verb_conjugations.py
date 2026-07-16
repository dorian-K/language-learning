import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from conjugation_validation import repair_conjugation_card, validate_conjugation_card
from llm import invoke_llm

VOCAB_SOURCE = os.getenv("VOCAB_SOURCE", "Refold ES1K")

# The LLM occasionally returns a malformed sentence (no [infinitive] blank, verb missing, …).
# We repair what we can and retry the whole call when a card is still invalid, so bad cards never
# reach the deck. Tunable; 0 disables retries (repair-only).
MAX_GEN_ATTEMPTS = int(os.getenv("MAX_GEN_ATTEMPTS", "3"))

VERBS_FILES = {
    "irregular": os.path.join(
        os.path.dirname(__file__), "../extra_vocab/simple_and_irregular_verbs.txt"
    ),
    "regular": os.path.join(os.path.dirname(__file__), "../extra_vocab/regular_verbs.txt"),
}
VOCAB_FOLDER = os.path.join(os.path.dirname(__file__), f"../anki/{VOCAB_SOURCE}")
OUTPUT_CONFIGS = {
    "irregular": {
        "folder": os.path.join(os.path.dirname(__file__), "../anki/irregular_verbs"),
        "deck_name": "Conjugations",
    },
    "regular": {
        "folder": os.path.join(os.path.dirname(__file__), "../anki/regular_verbs"),
        "deck_name": "Regular Conjugations",
    },
}
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "verb_conjugation_prompt.txt")

MAX_CONCURRENT_CALLS = 50

TENSES = [
    ("indicativo", "presente"),
    ("indicativo", "pretérito_indefinido"),
    ("indicativo", "pretérito_imperfecto"),
    ("indicativo", "futuro"),
    ("indicativo", "condicional"),
    ("subjuntivo", "presente"),
    ("subjuntivo", "imperfecto"),
    ("imperativo", "afirmativo"),
    ("imperativo", "negativo"),
]

PERSONS = [
    "yo",
    "tú",
    "él/ella/usted",
    "nosotros/nosotras",
    "vosotros/vosotras",
    "ellos/ellas/ustedes",
]

# The imperative mood has no first-person-singular form; we only drill the informal commands.
IMPERATIVE_PERSONS = ["tú", "vosotros/vosotras"]


def persons_for(tense_category):
    """Grammatical persons that exist for a given tense/mood."""
    return IMPERATIVE_PERSONS if tense_category == "imperativo" else PERSONS


def load_vocab_words():
    vocab = []
    spanish_words = set()
    if not os.path.exists(VOCAB_FOLDER):
        print(f"Warning: Vocab folder {VOCAB_FOLDER} not found.")
        return vocab

    for filename in os.listdir(VOCAB_FOLDER):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(VOCAB_FOLDER, filename)
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            for card in data:
                level = card.get("mandatory_level", "")
                if level in ["A1", "A2"]:
                    en = card.get("cue_en") or card.get("target_en", "")
                    de = card.get("cue_de") or card.get("target_de", "")
                    es = card.get("cue_spanish") or card.get("target_es", "")
                    if isinstance(en, list):
                        en = en[0] if en else ""
                    if isinstance(de, list):
                        de = de[0] if de else ""
                    if isinstance(es, list):
                        es = es[0] if es else ""
                    if en and de and es:
                        if es in spanish_words:
                            continue
                        spanish_words.add(es)
                        vocab.append({"en": en, "de": de, "es": es})
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    random.shuffle(vocab)
    return vocab


def load_verbs(filepath):
    verbs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("(")
            if len(parts) >= 2:
                infinitive = parts[0].strip()
                meaning = parts[1].replace(")", "").strip()
                verbs.append((infinitive, meaning))
            else:
                verbs.append((line.strip(), ""))
    return verbs


def build_llm_input(verb, tense_category, tense_name, person, vocab_sample):
    vocab_str = "\n".join([f"- {v['es']} ({v['en']}, {v['de']})" for v in vocab_sample])
    return f"""Verb: {verb}
Tense: {tense_category}/{tense_name}
Person: {person}

Vocabulary words to use in the sentence (A1-B1 level, use at least 2):
{vocab_str}
"""


def canonicalize_cards(cards, verb, tense_category, tense_name, person):
    """Overwrite the deterministic fields with the known inputs.

    The LLM echoes tense/person/infinitive inconsistently ("futuro" vs "indicativo/futuro" vs
    "pretérito_indefinido", "Ser" vs "ser", …). make_anki_deck derives each card's GUID and dedup
    key from direction|infinitive|tense|person, so an inconsistent echo yields unstable GUIDs —
    broken dedup and reset Anki scheduling on re-import. We already know the true values, so force
    them into the canonical form ("category/name", lowercase infinitive) used by the deck.
    """
    canonical_tense = f"{tense_category}/{tense_name}"
    canonical_infinitive = verb.strip().lower()
    for card in cards:
        if isinstance(card, dict):
            card["infinitive"] = canonical_infinitive
            card["tense"] = canonical_tense
            card["person"] = person
    return cards


def process_conjugation(
    verb,
    tense_category,
    tense_name,
    person,
    vocab_sample,
    prompt_text,
    output_folder,
    verbose=False,
):
    time.sleep(0.1 * random.random())
    key = f"{verb}_{tense_category}_{tense_name}_{person}".replace("/", "_")
    output_file = os.path.join(output_folder, f"{key}.json")

    if os.path.exists(output_file):
        print(f"Already processed: {key}, skipping.")
        return True

    llm_input = build_llm_input(verb, tense_category, tense_name, person, vocab_sample)
    user_message = f"{prompt_text}\n{llm_input}"

    try:
        vocab_data, issues = generate_valid_cards(
            verb, tense_category, tense_name, person, user_message, key
        )
        if issues:
            # Repair couldn't salvage it and retries were exhausted; don't write a broken card —
            # leaving no file means the next run regenerates it.
            print(f"WARNING: {key} still invalid after {MAX_GEN_ATTEMPTS} attempts: {issues}")
            return False

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=4, ensure_ascii=False)

        print(f"Generated: {key} ({len(vocab_data)} cards)")
        if verbose:
            print(json.dumps(vocab_data, indent=4, ensure_ascii=False))
        return True

    except Exception as e:
        print(f"Error processing {key}: {e}")
        return False


def generate_valid_cards(verb, tense_category, tense_name, person, user_message, key=""):
    """Invoke the LLM, canonicalize + repair, and retry until the cards validate.

    Returns ``(cards, issues)``; ``issues`` is empty when the cards are valid. On exhausted retries
    it returns the last attempt together with its remaining issues so the caller can decide.
    """
    cards, issues = [], ["no attempt made"]
    for attempt in range(1, MAX_GEN_ATTEMPTS + 1):
        cards = invoke_llm(
            [
                {
                    "role": "system",
                    "content": "You are an expert linguistics AI and Spanish language teacher. You output strict JSON arrays.",
                },
                {"role": "user", "content": user_message},
            ],
            print_reasoning=False,
        )
        cards = canonicalize_cards(cards, verb, tense_category, tense_name, person)
        for card in cards:
            repair_conjugation_card(card)
        issues = [issue for card in cards for issue in validate_conjugation_card(card)]
        if not issues:
            return cards, []
        print(f"  [retry {attempt}/{MAX_GEN_ATTEMPTS}] {key}: {issues}")
    return cards, issues


def main():
    for verb_type, verb_config in VERBS_FILES.items():
        output_folder = OUTPUT_CONFIGS[verb_type]["folder"]
        os.makedirs(output_folder, exist_ok=True)

        verbs = load_verbs(verb_config)
        vocab = load_vocab_words()

        print(f"[{verb_type}] Loaded {len(verbs)} verbs and {len(vocab)} vocabulary words.")

        if not vocab:
            print("Warning: No vocabulary words loaded, using defaults.")
            vocab = [
                {"en": "the house", "de": "das Haus", "es": "la casa"},
                {"en": "to eat", "de": "essen", "es": "comer"},
                {"en": "water", "de": "Wasser", "es": "el agua"},
            ]

        with open(PROMPT_FILE, encoding="utf-8") as f:
            prompt_text = f.read()

        is_sequential = os.getenv("SEQUENTIAL", "false").lower() in ("true", "1", "yes")
        is_dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")

        if is_dry_run:
            infinitive, _meaning = random.choice(verbs)
            tense_category, tense_name = random.choice(TENSES)
            person = random.choice(persons_for(tense_category))
            vocab_sample = random.sample(vocab, min(5, len(vocab)))

            print(f"\n=== DRY RUN [{verb_type}] ===")
            print(f"Verb: {infinitive}")
            print(f"Tense: {tense_category}/{tense_name}")
            print(f"Person: {person}")
            print(f"Vocab sample: {[v['es'] for v in vocab_sample]}")
            print()

            llm_input = build_llm_input(
                infinitive, tense_category, tense_name, person, vocab_sample
            )
            user_message = f"{prompt_text}\n{llm_input}"

            vocab_data = invoke_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert linguistics AI and Spanish teacher. You output strict JSON arrays.",
                    },
                    {"role": "user", "content": user_message},
                ],
                print_reasoning=False,
            )
            vocab_data = canonicalize_cards(
                vocab_data, infinitive, tense_category, tense_name, person
            )

            print("--- LLM output (formatted) ---")
            print(json.dumps(vocab_data, indent=4, ensure_ascii=False))
            print()
            continue

        tasks = []
        for infinitive, _meaning in verbs:
            for tense_category, tense_name in TENSES:
                for person in persons_for(tense_category):
                    vocab_sample = random.sample(vocab, min(5, len(vocab)))
                    tasks.append((infinitive, tense_category, tense_name, person, vocab_sample))

        print(f"[{verb_type}] Total tasks: {len(tasks)}")

        if is_sequential:
            print(f"[{verb_type}] Running sequentially (no parallelization)")
            for infinitive, tense_category, tense_name, person, vocab_sample in tasks:
                process_conjugation(
                    infinitive,
                    tense_category,
                    tense_name,
                    person,
                    vocab_sample,
                    prompt_text,
                    output_folder,
                    verbose=True,
                )
        else:
            print(f"[{verb_type}] Parallelization with {MAX_CONCURRENT_CALLS}")
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
                futures = {
                    executor.submit(
                        process_conjugation,
                        infinitive,
                        tense_category,
                        tense_name,
                        person,
                        vocab_sample,
                        prompt_text,
                        output_folder,
                        verbose=False,
                    ): (infinitive, tense_category, tense_name, person)
                    for infinitive, tense_category, tense_name, person, vocab_sample in tasks
                }

                for future in as_completed(futures):
                    future.result()


if __name__ == "__main__":
    main()
