from spoken_word import spoken_word


def test_spanish_to_target_uses_cue():
    card = {"direction": "spanish_to_target", "cue_spanish": "levantar", "target_es": ["ignored"]}
    assert spoken_word(card) == "levantar"


def test_spanish_sentence_to_target_alias():
    card = {"direction": "spanish_sentence_to_target", "cue_spanish": "la manzana"}
    assert spoken_word(card) == "la manzana"


def test_target_to_spanish_uses_first_target():
    card = {"direction": "target_to_spanish", "target_es": ["levantar", "alzar"]}
    assert spoken_word(card) == "levantar"


def test_target_sentence_to_spanish_alias():
    card = {"direction": "target_sentence_to_spanish", "target_es": ["comer"]}
    assert spoken_word(card) == "comer"


def test_both_directions_of_same_word_match():
    # The clip is content-hash-keyed on the spoken text, so both directions must agree.
    s2t = {"direction": "spanish_to_target", "cue_spanish": "levantar"}
    t2s = {"direction": "target_to_spanish", "target_es": ["levantar", "alzar"]}
    assert spoken_word(s2t) == spoken_word(t2s)


def test_target_es_as_plain_string():
    card = {"direction": "target_to_spanish", "target_es": "correr"}
    assert spoken_word(card) == "correr"


def test_empty_and_missing():
    assert spoken_word({}) == ""
    assert spoken_word({"direction": "spanish_to_target", "cue_spanish": None}) == ""
    assert spoken_word({"direction": "target_to_spanish", "target_es": []}) == ""
    assert spoken_word({"direction": "numbers"}) == ""
