import argparse, logging
from collections import Counter

from hanzidentifier import SIMPLIFIED, TRADITIONAL, identify
import unicodedataplus as ud

LOGGER = logging.getLogger(__package__ + ".script_utils")

SCRIPT_CODES = {
    "Adlam": "Adlm",
    "Caucasian_Albanian": "Aghb",
    "Ahom": "Ahom",
    "Arabic": "Arab",
    "Imperial_Aramaic": "Armi",
    "Armenian": "Armn",
    "Avestan": "Avst",
    "Balinese": "Bali",
    "Bamum": "Bamu",
    "Bassa_Vah": "Bass",
    "Batak": "Batk",
    "Bengali": "Beng",
    "Bhaiksuki": "Bhks",
    "Bopomofo": "Bopo",
    "Brahmi": "Brah",
    "Braille": "Brai",
    "Buginese": "Bugi",
    "Buhid": "Buhd",
    "Chakma": "Cakm",
    "Canadian_Aboriginal": "Cans",
    "Carian": "Cari",
    "Cham": "Cham",
    "Cherokee": "Cher",
    "Chorasmian": "Chrs",
    "Coptic": "Copt",
    "Cypro_Minoan": "Cpmn",
    "Cypriot": "Cprt",
    "Cyrillic": "Cyrl",
    "Devanagari": "Deva",
    "Dives_Akuru": "Diak",
    "Dogra": "Dogr",
    "Deseret": "Dsrt",
    "Duployan": "Dupl",
    "Egyptian_Hieroglyphs": "Egyp",
    "Elbasan": "Elba",
    "Elymaic": "Elym",
    "Ethiopic": "Ethi",
    "Georgian": "Geor",
    "Glagolitic": "Glag",
    "Gunjala_Gondi": "Gong",
    "Masaram_Gondi": "Gonm",
    "Gothic": "Goth",
    "Grantha": "Gran",
    "Greek": "Grek",
    "Gujarati": "Gujr",
    "Gurmukhi": "Guru",
    "Hangul": "Hang",
    "Han": "Hani",
    "Hanunoo": "Hano",
    "Hatran": "Hatr",
    "Hebrew": "Hebr",
    "Hiragana": "Hira",
    "Anatolian_Hieroglyphs": "Hluw",
    "Pahawh_Hmong": "Hmng",
    "Nyiakeng_Puachue_Hmong": "Hmnp",
    "Katakana_Or_Hiragana": "Hrkt",
    "Old_Hungarian": "Hung",
    "Old_Italic": "Ital",
    "Javanese": "Java",
    "Kayah_Li": "Kali",
    "Katakana": "Kana",
    "Kawi": "Kawi",
    "Kharoshthi": "Khar",
    "Khmer": "Khmr",
    "Khojki": "Khoj",
    "Khitan_Small_Script": "Kits",
    "Kannada": "Knda",
    "Kaithi": "Kthi",
    "Tai_Tham": "Lana",
    "Lao": "Laoo",
    "Latin": "Latn",
    "Lepcha": "Lepc",
    "Limbu": "Limb",
    "Linear_A": "Lina",
    "Linear_B": "Linb",
    "Lisu": "Lisu",
    "Lycian": "Lyci",
    "Lydian": "Lydi",
    "Mahajani": "Mahj",
    "Makasar": "Maka",
    "Mandaic": "Mand",
    "Manichaean": "Mani",
    "Marchen": "Marc",
    "Medefaidrin": "Medf",
    "Mende_Kikakui": "Mend",
    "Meroitic_Cursive": "Merc",
    "Meroitic_Hieroglyphs": "Mero",
    "Malayalam": "Mlym",
    "Modi": "Modi",
    "Mongolian": "Mong",
    "Mro": "Mroo",
    "Meetei_Mayek": "Mtei",
    "Multani": "Mult",
    "Myanmar": "Mymr",
    "Nag_Mundari": "Nagm",
    "Nandinagari": "Nand",
    "Old_North_Arabian": "Narb",
    "Nabataean": "Nbat",
    "Newa": "Newa",
    "Nko": "Nkoo",
    "Nushu": "Nshu",
    "Ogham": "Ogam",
    "Ol_Chiki": "Olck",
    "Old_Turkic": "Orkh",
    "Oriya": "Orya",
    "Osage": "Osge",
    "Osmanya": "Osma",
    "Old_Uyghur": "Ougr",
    "Palmyrene": "Palm",
    "Pau_Cin_Hau": "Pauc",
    "Old_Permic": "Perm",
    "Phags_Pa": "Phag",
    "Inscriptional_Pahlavi": "Phli",
    "Psalter_Pahlavi": "Phlp",
    "Phoenician": "Phnx",
    "Miao": "Plrd",
    "Inscriptional_Parthian": "Prti",
    "Rejang": "Rjng",
    "Hanifi_Rohingya": "Rohg",
    "Runic": "Runr",
    "Samaritan": "Samr",
    "Old_South_Arabian": "Sarb",
    "Saurashtra": "Saur",
    "SignWriting": "Sgnw",
    "Shavian": "Shaw",
    "Sharada": "Shrd",
    "Siddham": "Sidd",
    "Khudawadi": "Sind",
    "Sinhala": "Sinh",
    "Sogdian": "Sogd",
    "Old_Sogdian": "Sogo",
    "Sora_Sompeng": "Sora",
    "Soyombo": "Soyo",
    "Sundanese": "Sund",
    "Syloti_Nagri": "Sylo",
    "Syriac": "Syrc",
    "Tagbanwa": "Tagb",
    "Takri": "Takr",
    "Tai_Le": "Tale",
    "New_Tai_Lue": "Talu",
    "Tamil": "Taml",
    "Tangut": "Tang",
    "Tai_Viet": "Tavt",
    "Telugu": "Telu",
    "Tifinagh": "Tfng",
    "Tagalog": "Tglg",
    "Thaana": "Thaa",
    "Thai": "Thai",
    "Tibetan": "Tibt",
    "Tirhuta": "Tirh",
    "Tangsa": "Tnsa",
    "Toto": "Toto",
    "Ugaritic": "Ugar",
    "Vai": "Vaii",
    "Vithkuqi": "Vith",
    "Warang_Citi": "Wara",
    "Wancho": "Wcho",
    "Old_Persian": "Xpeo",
    "Cuneiform": "Xsux",
    "Yezidi": "Yezi",
    "Yi": "Yiii",
    "Zanabazar_Square": "Zanb",
    "Inherited": "Zinh",
    "Common": "Zyyy",
    "Unknown": "Zzzz",
}

REPRESENTED_SCRIPTS = {
    "facebook/nllb-200": [
        "Arab",
        "Latn",
        "Ethi",
        "Beng",
        "Deva",
        "Cyrl",
        "Tibt",
        "Grek",
        "Gujr",
        "Hebr",
        "Armn",
        "Jpan",
        "Knda",
        "Geor",
        "Khmr",
        "Hang",
        "Laoo",
        "Mlym",
        "Mymr",
        "Orya",
        "Guru",
        "Sinh",
        "Telu",
        "Thai",
        "Taml",
        "Tfng",
        "Hant",
        "Hans",
    ],
    "google/madlad400": [
        "Latn",
        "Cyrl",
        "Hans",
        "Hant",
        "Jpan",
        "Thai",
        "Arab",
        "Grek",
        "Hang",
        "Kore",
        "Hebr",
        "Deva",
        "Taml",
        "Mlym",
        "Telu",
        "Beng",
        "Geor",
        "Knda",
        "Gujr",
        "Sinh",
        "Armn",
        "Mymr",
        "Khmr",
        "Laoo",
        "Ethi",
        "Orya",
        "Tibt",
        "Syrc",
        "Cher",
        "Tfng",
        "Thaa",
        "Guru",
        "Cans",
    ],
}


def script(char):
    return ud.script(char)


def predict_han_variant(text):
    num_trad, num_simp = 0, 0
    for c in text:
        ct = identify(c)
        num_trad += ct == TRADITIONAL
        num_simp += ct == SIMPLIFIED
    return "Hant" if num_trad > num_simp else "Hans"


def predict_script_code(text):
    "Predict the ISO 15924 script code for the dominant script in text"
    if len(text) == 0:
        return "None"
    counts = Counter([script(char) for char in text])
    pred = counts.most_common()[0][0]
    if pred in ("Hiragana", "Katakana") or (pred == "Han" and (counts.get("Hiragana") or counts.get("Katakana"))):
        return "Jpan"
    if pred == "Han":
        return predict_han_variant(text)
    return SCRIPT_CODES[pred]


def is_represented(script_code, model):
    "Check if a script code is represented in the given model"
    return any(model.startswith(pfx) and script_code in scripts for pfx, scripts in REPRESENTED_SCRIPTS.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The text file you want to determine the script of")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8-sig") as f:
        text = f.read()
    file_script = predict_script_code(text)
    LOGGER.info(f"Script: {file_script}")


if __name__ == "__main__":
    main()
