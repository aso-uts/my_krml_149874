import pytest
from my_krml_149874 import my_krml_149874

def capital_case(x):
    return x.capitalize()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'