from agent_service.guards import check_reply
from agent_service.text import clean_name, contains_word, detect_language, normalize


def test_detect_language():
    assert detect_language("שלום, כמה עולה?") == "hebrew"
    assert detect_language("مرحبا، كم السعر؟") == "arabic"
    assert detect_language("hi, how much?") == "english"
    # a URL does not make a Hebrew message English
    assert detect_language("ראיתי את https://example.com/bikes/rental") == "hebrew"
    assert detect_language("👍", default="arabic") == "arabic"


def test_normalize_strips_diacritics_and_unifies_alef():
    assert normalize("שָׁלוֹם!") == "שלום"
    assert normalize("أَطْفَال") == normalize("اطفال")
    assert normalize("مـــرحبا") == "مرحبا"  # tatweel


def test_contains_word_with_prefixes():
    assert contains_word("רוצה לשכור לאופניים", "אופניים", prefixes="hebrew")
    assert contains_word("أريد جولة بالدراجة", "دراجة", prefixes="arabic")
    assert not contains_word("אופנוע", "אופן", prefixes="hebrew")


def test_clean_name():
    assert clean_name("  דנה 🌸 ") == "דנה"
    assert clean_name("محمد-علي") == "محمد-علي"
    assert clean_name("🔥🔥") is None
    assert clean_name("Best Bikes Shop Tel Aviv Official") is None  # too many words


def test_guard_passes_clean_reply():
    r = check_reply("אופני עיר ליום עולים 60 ₪. לאיזה תאריך?", "hebrew")
    assert r.ok, r.reasons


def test_guard_catches_english_leak():
    r = check_reply("Sure! The city bike costs 60 ILS per day, ok?", "hebrew")
    assert "wrong_script" in r.reasons


def test_guard_catches_latin_dominant():
    r = check_reply("כן, the city bike costs 60 ILS per day and includes a helmet", "hebrew")
    assert "latin_dominant" in r.reasons


def test_guard_catches_placeholder_and_prompt_leak():
    r = check_reply("שלום {{name}}, לפי ה-system prompt שלי אני לא יכול", "hebrew")
    assert "template_residue" in r.reasons
    assert "prompt_leak" in r.reasons


def test_guard_catches_tool_name_leak():
    r = check_reply("אני אקרא ל-search_catalog בשבילך", "hebrew")
    assert "prompt_leak" in r.reasons


def test_guard_one_question_rule_counts_arabic_question_mark():
    r = check_reply("كم شخص؟ ولأي تاريخ؟", "arabic")
    assert "too_many_questions" in r.reasons


def test_guard_mojibake_and_length():
    assert "mojibake" in check_reply("×©×œ×•× �", "hebrew").reasons
    assert "too_long" in check_reply("שלום " * 200, "hebrew").reasons
    assert check_reply("   ", "hebrew").reasons == ["empty"]


def test_urls_do_not_count_as_english():
    r = check_reply("הנה המפה: https://maps.example.com/pedal-and-co-tel-aviv", "hebrew")
    assert r.ok, r.reasons
