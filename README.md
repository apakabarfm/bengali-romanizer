[![Tests](https://github.com/apakabarfm/bengali-romanizer/actions/workflows/test.yml/badge.svg)](https://github.com/apakabarfm/bengali-romanizer/actions/workflows/test.yml)

# Bengali Romanizer

A simple library for romanizing Bengali text to Latin script.

## Installation

```bash
pip install git+https://github.com/apakabarfm/bengali-romanizer
```

## Usage

```python
>>> import bengali_romanizer
>>> bengali_romanizer.romanize('বাংলা')
'bāṅlā'
>>> bengali_romanizer.romanize('নমস্কার')
'namskār'
>>> bengali_romanizer.romanize('ধন্যবাদ')
'dhnyabād'
>>> bengali_romanizer.romanize('ভক্তি')
'bhakti'
>>> bengali_romanizer.romanize('আন্দোলন')
'āndôln'
>>> bengali_romanizer.romanize('প্রাচীন')
'prācīn'
```

## Features

- Accurate romanization of Bengali text
- Handles complex Bengali orthography including:
  - Conjunct consonants (যুক্তাক্ষর)
  - Vowel signs and diacritics
  - Special marks (anusvara, visarga, chandrabindu)
  - Nukta consonants
- Based on linguistic analysis of Bengali phonology
- Comprehensive test coverage