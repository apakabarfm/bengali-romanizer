import pytest
import yaml
from pathlib import Path
from bengali_romanizer import romanize


def load_test_cases():
    """Load test cases from YAML file"""
    yaml_file = Path(__file__).parent / "test_cases.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("test_case", [
    case for category in load_test_cases().values() 
    for case in category
])
def test_yaml_cases(test_case):
    """Test Bengali romanization using YAML test cases"""
    if 'skip' in test_case:
        pytest.skip(test_case['skip'])
    
    result = romanize(test_case['input'])
    assert result == test_case['expected'], f"Test: {test_case['title']}"