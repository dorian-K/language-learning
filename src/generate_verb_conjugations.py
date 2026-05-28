import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import invoke_llm

VOCAB_SOURCE = os.getenv("VOCAB_SOURCE", "Refold ES1K")

VERBS_FILE = os.path.join(
    os.path.dirname(__file__), "../extra_vocab/simple_and_irregular_verbs.txt"
)
VOCAB_FOLDER = os.path.join(os.path.dirname(__file__), f"../anki/{VOCAB_SOURCE}")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../anki/irregular_verbs")
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "verb_conjugation_prompt.txt")

MAX_CONCURRENT_CALLS = 10

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


def load_verbs():
    verbs = []
    with open(VERBS_FILE, encoding="utf-8") as f:
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


def process_conjugation(
    verb, tense_category, tense_name, person, vocab_sample, prompt_text, verbose=False
):
    time.sleep(0.1 * random.random())
    key = f"{verb}_{tense_category}_{tense_name}_{person}".replace("/", "_")
    output_file = os.path.join(OUTPUT_FOLDER, f"{key}.json")

    if os.path.exists(output_file):
        print(f"Already processed: {key}, skipping.")
        return True

    llm_input = build_llm_input(verb, tense_category, tense_name, person, vocab_sample)
    user_message = f"{prompt_text}\n{llm_input}"

    try:
        vocab_data = invoke_llm(
            [
                {
                    "role": "system",
                    "content": "You are an expert linguistics AI and Spanish language teacher. You output strict JSON arrays.",
                },
                {"role": "user", "content": user_message},
            ],
            print_reasoning=False,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=4, ensure_ascii=False)

        print(f"Generated: {key} ({len(vocab_data)} cards)")
        if verbose:
            print(json.dumps(vocab_data, indent=4, ensure_ascii=False))
        return True

    except Exception as e:
        print(f"Error processing {key}: {e}")
        return False


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    verbs = load_verbs()
    vocab = load_vocab_words()

    print(f"Loaded {len(verbs)} verbs and {len(vocab)} vocabulary words.")

    if not vocab:
        print("Warning: No vocabulary words loaded, using defaults.")
        vocab = [
            {"en": "the house", "de": "das Haus", "es": "la casa"},
            {"en": "to eat", "de": "essen", "es": "comer"},
            {"en": "water", "de": "Wasser", "es": "el agua"},
        ]
        raise RuntimeError()

    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompt_text = f.read()

    is_sequential = os.getenv("SEQUENTIAL", "false").lower() in ("true", "1", "yes")
    is_dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")

    if is_dry_run:
        infinitive, _meaning = random.choice(verbs)
        tense_category, tense_name = random.choice(TENSES)
        person = random.choice(PERSONS)
        vocab_sample = random.sample(vocab, min(5, len(vocab)))

        print("\n=== DRY RUN ===")
        print(f"Verb: {infinitive}")
        print(f"Tense: {tense_category}/{tense_name}")
        print(f"Person: {person}")
        print(f"Vocab sample: {[v['es'] for v in vocab_sample]}")
        print()

        llm_input = build_llm_input(infinitive, tense_category, tense_name, person, vocab_sample)
        user_message = f"{prompt_text}\n{llm_input}"

        # print("--- LLM input ---")
        # print(user_message)
        print()

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

        print("--- LLM output (formatted) ---")
        print(json.dumps(vocab_data, indent=4, ensure_ascii=False))
        print()
        return

    tasks = []
    for infinitive, _meaning in verbs:
        for tense_category, tense_name in TENSES:
            for person in PERSONS:
                if tense_category == "imperativo" and person not in [
                    "tú",
                    "vosotros/vosotras",
                ]:
                    continue
                vocab_sample = random.sample(vocab, min(5, len(vocab)))
                tasks.append((infinitive, tense_category, tense_name, person, vocab_sample))

    print(f"Total tasks: {len(tasks)}")

    if is_sequential:
        print("Running sequentially (no parallelization)")
        for infinitive, tense_category, tense_name, person, vocab_sample in tasks:
            process_conjugation(
                infinitive,
                tense_category,
                tense_name,
                person,
                vocab_sample,
                prompt_text,
                verbose=True,
            )
    else:
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
                    verbose=False,
                ): (infinitive, tense_category, tense_name, person)
                for infinitive, tense_category, tense_name, person, vocab_sample in tasks
            }

            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
