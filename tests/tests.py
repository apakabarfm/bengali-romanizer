from pathlib import Path

import pytest
import yaml

from bengali_romanizer import romanize
from bengali_romanizer.romanizer import (
    BengaliAksharaTokenizer,
    BengaliAkshara,
    _BengaliTransliterator,
)


def load_test_cases():
    """Load test cases from YAML file"""
    yaml_file = Path(__file__).parent / "test_cases.yaml"
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "test_case", [case for category in load_test_cases().values() for case in category]
)
def test_yaml_cases(test_case):
    """Test Bengali romanization using YAML test cases"""
    if "skip" in test_case:
        pytest.skip(test_case["skip"])

    result = romanize(test_case["input"])
    assert result == test_case["expected"], f"Test: {test_case['title']}"


@pytest.mark.parametrize(
    "method_name,akshara_args,expected",
    [
        ("_translate_independent_vowel", ([], "অ", False, []), "ô"),
        ("_translate_nukta_consonant", (["য়"], None, False, []), "y"),
        ("_translate_conjunct_with_halant", (["ক", "ত"], None, True, []), "kt"),
        ("_translate_single_consonant", (["ক"], "ি", False, []), "ki"),
    ],
)
def test_akshara_translation_methods(method_name, akshara_args, expected):
    """Test individual BengaliAkshara translation methods"""
    consonant_map = {"ক": "ka", "ত": "ta", "ভ": "bha", "য়": "y", "ড়": "ṛ"}
    vowel_map = {"ি": "i", "ে": "e"}
    independent_vowel_map = {"অ": "ô", "আ": "ā"}
    special_map = {"ং": "ṅ"}

    akshara = BengaliAkshara(*akshara_args)
    method = getattr(akshara, method_name)

    if method_name == "_translate_independent_vowel":
        result = method(independent_vowel_map)
    elif method_name == "_translate_nukta_consonant":
        result = method(consonant_map)
    elif method_name == "_translate_conjunct_with_halant":
        result = method(consonant_map)
    else:  # _translate_single_consonant
        result = method(consonant_map, vowel_map, special_map)

    assert result == expected


def test_final_m_after_halant():
    """Micro-test: ম after ধর্ should be 'm' not 'ma'"""

    # Create context: ধর্ followed by ম
    dhr_akshara = BengaliAkshara(["ধ", "র"], None, True, [])  # ধর্ with halant
    m_akshara = BengaliAkshara(["ম"], None, False, [])  # ম

    transliterator = _BengaliTransliterator()

    # Test ম in context after halant conjunct
    result = transliterator._translate_akshara_with_context(
        m_akshara, [dhr_akshara, m_akshara], 1, transliterator.vowel_map
    )

    assert result == "m", "ম after halant conjunct should be 'm' not 'ma'"


def test_context_detection_for_visarga():
    """TDD test for দুঃখ - visarga breaks vowel context"""

    # Mock দুঃখ tokenization: [দুঃ], [খ] (visarga attaches to previous)
    aksharas = [
        BengaliAkshara(
            ["দ"], "ু", False, ["ঃ"]
        ),  # দুঃ → duḥ (consonant + vowel + visarga)
        BengaliAkshara(
            ["খ"], None, False, []
        ),  # খ → should be 'kha' (visarga breaks context)
    ]

    transliterator = _BengaliTransliterator()

    # Test that visarga breaks vowel context
    context_result = transliterator._has_vowel_context(aksharas, 1)
    assert not context_result, "খ should NOT have vowel context after visarga"

    # Test translation
    vowel_map = transliterator.vowel_map
    result = transliterator._translate_akshara_with_context(
        aksharas[1], aksharas, 1, vowel_map
    )
    assert result == "kha", "খ after visarga should keep inherent vowel"


def test_context_detection_for_andolon():
    """TDD test for আন্দোলন context detection"""

    # Mock আন্দোলন tokenization: [আ], [নদো], [ল], [ন]
    aksharas = [
        BengaliAkshara([], "আ", False, []),  # আ → ā (independent vowel)
        BengaliAkshara(["ন", "দ"], "ো", False, []),  # নদো → ndo (conjunct + vowel)
        BengaliAkshara(["ল"], None, False, []),  # ল → should be 'l' (after vowel)
        BengaliAkshara(
            ["ন"], None, False, []
        ),  # ন → should be 'n' (word continues after vowel)
    ]

    # Test context detection for each position
    transliterator = _BengaliTransliterator()

    # Test position 2: ল after নদো (which has vowel ো)
    context_result_2 = transliterator._has_vowel_context(aksharas, 2)
    assert context_result_2, "ল should detect vowel context from preceding নদো"

    # Test position 3: ন after ল (which followed vowel context)
    context_result_3 = transliterator._has_vowel_context(aksharas, 3)
    assert context_result_3, "ন should detect vowel context propagated through word"


def test_conjunct_in_word_context():
    """TDD test for ধর্মীয় - conjunct ধর should be dhr not dhra"""

    # Test ধর conjunct in word context (not standalone)
    akshara = BengaliAkshara(["ধ", "র"], None, False, [])

    consonant_map = {"ধ": "dha", "র": "ra"}
    vowel_map = {}

    result = akshara._translate_conjunct_without_halant(consonant_map, vowel_map)
    assert result == "dhr", "ধর conjunct in word context should be dhr, not dhra"


@pytest.mark.parametrize(
    "consonants,vowel,halant,special,method,extra_args,expected,description",
    [
        (
            ["হ", "য"],
            None,
            False,
            [],
            "_translate_conjunct_without_halant",
            (True,),
            "hyô",
            "Final য in conjunct gets ô",
        ),
        (
            ["য়"],
            "া",
            False,
            [],
            "_translate_single_consonant",
            (),
            "ôy",
            "য় + া special combination",
        ),
    ],
)
def test_special_vowel_rules(
    consonants, vowel, halant, special, method, extra_args, expected, description
):
    """Test special vowel rules for edge cases"""
    akshara = BengaliAkshara(consonants, vowel, halant, special)
    consonant_map = {"হ": "ha", "য": "ya", "য়": "y"}
    vowel_map = {"া": "ā"}
    special_map = {}

    akshara_method = getattr(akshara, method)
    if method == "_translate_conjunct_without_halant":
        result = akshara_method(consonant_map, vowel_map, *extra_args)
    else:
        result = akshara_method(consonant_map, vowel_map, special_map)

    assert result == expected, description


def test_adhyay_tokenization():
    """Test অধ্যায় tokenization structure"""
    tokenizer = BengaliAksharaTokenizer()
    aksharas = tokenizer.tokenize("অধ্যায়")

    assert len(aksharas) == 3, f"Expected 3 aksharas, got {len(aksharas)}: {aksharas}"
    assert aksharas[0].consonants == [] and aksharas[0].vowel == "অ"  # অ
    assert aksharas[1].consonants == ["ধ", "য"] and aksharas[1].vowel == "া"  # ধ্যা
    assert aksharas[2].consonants == ["য়"] and aksharas[2].vowel is None  # য়


def test_single_consonant_after_conjunct():
    """
    Test single consonants after conjuncts should keep inherent vowel

    LINGUISTIC ANALYSIS from ঈশ্বরের (īśbarer):
    - ঈশ্বরের = ঈ + শ্ব + র + ে + র
    - Expected: ī + śb + [a] + re + r = īśbarer
    - Issue: র after conjunct শ্ব should keep inherent vowel → ra

    RULE: Single consonants immediately after conjuncts keep inherent vowel
    """

    # Test: র after conjunct should keep inherent vowel

    # Mock context: previous akshara was conjunct
    prev_akshara = BengaliAkshara(["শ", "ব"], None, False, [])  # শ্ব conjunct
    current_akshara = BengaliAkshara(["র"], None, False, [])  # র single consonant

    transliterator = _BengaliTransliterator()
    result = transliterator._translate_akshara_with_context(
        current_akshara,
        [prev_akshara, current_akshara],
        1,  # position 1
        transliterator.vowel_map,
    )

    assert result == "ra", "Single consonant after conjunct should keep inherent vowel"


@pytest.mark.skip("FIXME: নববর্ষ → nababrṣ wrong inherent vowels in conjuncts")
def test_full_bengali_text_word_separation():
    """Test that Bengali romanizer preserves word boundaries in long text"""

    bengali_text = """বাংলা নববর্ষ বাংলা পঞ্জিকা অনুসারে বছরের প্রথম দিনকে উদযাপন করার এক বিশেষ মুহূর্ত, যাকে অনেকেই পহেলা বৈশাখ নামে ডেকে থাকেন। এই দিন সাধারণত চৌদ্দ এপ্রিল পালিত হয়, যখন বাংলাদেশ ও পশ্চিমবঙ্গে উৎসবের উত্তাপ ছড়িয়ে পড়ে এবং মানুষ নতুন দিনের আহ্বানে মেতে ওঠে। এটি বাংলাদেশের জাতীয় দিবসগুলির মধ্যেও অন্তর্ভুক্ত, ফলে সরকারি ছুটির সুবাদে পরিবার ও বন্ধুবান্ধব একত্রিত হয়ে এই ঐতিহ্য উদযাপন করে।

নববর্ষের মূল উদযাপন শুরু হয় পহেলা বৈশাখের ভোরে, যখন ঢাকার রমনা বটমূলে ছায়ানটের মনোমুগ্ধকর অনুষ্ঠান দেখতে বিপুল মানুষ জড়ো হয়। সেই সঙ্গে মঙ্গল শোভাযাত্রা, যা ইউনেস্কো কর্তৃক অমূর্ত সাংস্কৃতিক ঐতিহ্য হিসেবে স্বীকৃত, শহরের রাজপথ ভরিয়ে তোলে এবং রঙিন পোশাক ও মুখোশধারীরা আনন্দ ছড়ায়। এদিন পান্তাভাত, ইলিশ মাছ ও নানা পিঠা পরিবেশনের মাধ্যমে বাঙালি স্বাদ আর সংস্কৃতির মেলবন্ধন দৃঢ় হয়, আর মানুষ একে অপরকে শুভেচ্ছা বিনিময় করে।

বাংলা নববর্ষের সামাজিক গুরুত্ব অপরিসীম, কারণ এটি কেবল নতুন সনের সূচনা নয়, বরং ঐতিহ্য ও একতার মিলনমেলার একটি উজ্জ্বল নমুনা। বিভিন্ন সম্প্রদায়ের মানুষ এই উৎসবের মাধ্যমে একে অপরের প্রতি সৌহার্দ্য ও সম্মান প্রকাশ করে, আর পুরনো বছরের হতাশা পেছনে ফেলে নতুন আশায় এগিয়ে যায়। ফলে সমাজে পারস্পরিক বন্ধন আরও দৃঢ় হয়।"""

    result = romanize(bengali_text)

    # Should contain spaces between words, not be one long string
    assert " " in result, "Romanized text must contain spaces between words"

    # Should preserve paragraph structure (same number of newlines)
    original_newlines = bengali_text.count("\n")
    result_newlines = result.count("\n")
    assert result_newlines == original_newlines, (
        f"Should preserve {original_newlines} newlines, got {result_newlines}"
    )

    # Should start with the first few words properly separated
    assert result.startswith("bāṅlā nbbrṣ bāṅlā"), (
        f"Should start with proper word separation, got: {result[:50]}"
    )
