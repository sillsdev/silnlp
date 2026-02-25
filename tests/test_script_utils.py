from silnlp.common.script_utils_ud import script, predict_han_variant, predict_script_code, is_represented, SCRIPT_CODES

def test_script():
    assert script('A') == 'Latin'
    assert script('α') == 'Greek'
    assert script('Б') == 'Cyrillic'
    assert script('अ') == 'Devanagari'
    assert script('あ') == 'Hiragana'
    assert script('ア') == 'Katakana'
    assert script('中') == 'Han'
    assert script('ก') == 'Thai'
    assert script('ア') == 'Katakana'
    assert script('한') == 'Hangul'
    assert script('ا') == 'Arabic'
    assert script('א') == 'Hebrew'
    assert script('ꯀ') == 'Meetei_Mayek'
    assert script(' ') == 'Common'
    assert script('0') == 'Common'
    assert script('\u0041') == 'Latin'   # 'A'
    assert script('\u10D0') == 'Georgian'
    assert script('\u0F40') == 'Tibetan'
    assert script('\u0E81') == 'Lao'
    assert script('\u1000') == 'Myanmar'
    assert script('\u0B85') == 'Tamil'
    assert script('\u0C05') == 'Telugu'
    assert script('\u0C85') == 'Kannada'
    assert script('\u0D05') == 'Malayalam'
    assert script('\u0D85') == 'Sinhala'
    assert script('\u1200') == 'Ethiopic'
    print('test_script passed')

def test_predict_han_variant():
    assert predict_han_variant('简体中文测试') == 'Hans'
    assert predict_han_variant('簡體中文測試') == 'Hant'
    assert predict_han_variant('') in ('Hans', 'Hant')
    print('test_predict_han_variant passed')

def test_predict_script_code():
    assert predict_script_code('') == 'None'
    assert predict_script_code('Hello world') == 'Latn'
    assert predict_script_code('Привет мир') == 'Cyrl'
    assert predict_script_code('こんにちは世界') == 'Jpan'
    assert predict_script_code('カタカナテスト') == 'Jpan'
    assert predict_script_code('简体中文测试文本') == 'Hans'
    assert predict_script_code('簡體中文測試文本') == 'Hant'
    assert predict_script_code('สวัสดีชาวโลก') == 'Thai'
    assert predict_script_code('नमस्ते दुनिया') == 'Deva'
    assert predict_script_code('مرحبا بالعالم') == 'Arab'
    assert predict_script_code('שלום עולם') == 'Hebr'
    assert predict_script_code('안녕하세요') == 'Hang'
    assert predict_script_code('Αλφα Βήτα') == 'Grek'
    assert predict_script_code('ამბავი') == 'Geor'
    assert predict_script_code('ꯃꯤꯇꯩ') == 'Mtei'
    assert predict_script_code('බුද්ධ') == 'Sinh'
    assert predict_script_code('తెలుగు') == 'Telu'
    assert predict_script_code('ខ្មែរ') == 'Khmr'
    assert predict_script_code('ລາວ') == 'Laoo'
    assert predict_script_code('မြန်မာ') == 'Mymr'
    assert predict_script_code('தமிழ்') == 'Taml'
    assert predict_script_code('ಕನ್ನಡ') == 'Knda'
    assert predict_script_code('മലയാളം') == 'Mlym'
    assert predict_script_code('ᓀᐦᐃᔭᐍᐏᐣ') == 'Cans'
    print('test_predict_script_code passed')

def test_predict_script_code_mixed():
    assert predict_script_code('Hello こんにちは world') == 'Latn'
    assert predict_script_code('Mostly Latin with one α') == 'Latn'
    print('test_predict_script_code_mixed passed')

def test_script_codes_mapping():
    assert SCRIPT_CODES['Latin'] == 'Latn'
    assert SCRIPT_CODES['Greek'] == 'Grek'
    assert SCRIPT_CODES['Cyrillic'] == 'Cyrl'
    assert SCRIPT_CODES['Han'] == 'Hani'
    assert SCRIPT_CODES['Arabic'] == 'Arab'
    assert SCRIPT_CODES['Hebrew'] == 'Hebr'
    assert SCRIPT_CODES['Devanagari'] == 'Deva'
    assert SCRIPT_CODES['Common'] == 'Zyyy'
    assert SCRIPT_CODES['Inherited'] == 'Zinh'
    print('test_script_codes_mapping passed')

def test_is_represented():
    assert is_represented('Latn', 'facebook/nllb-200-distilled-600M') == True
    assert is_represented('Arab', 'facebook/nllb-200-3.3B') == True
    assert is_represented('Latn', 'google/madlad400-3b-mt') == True
    assert is_represented('Syrc', 'google/madlad400-3b-mt') == True
    assert is_represented('Syrc', 'facebook/nllb-200-distilled-600M') == False
    assert is_represented('Latn', 'unknown/model') == False
    assert is_represented('Zzzz', 'facebook/nllb-200-3.3B') == False
    print('test_is_represented passed')

test_script()
test_predict_han_variant()
test_predict_script_code()
test_predict_script_code_mixed()
test_script_codes_mapping()
test_is_represented()
print('\nAll tests passed!')
