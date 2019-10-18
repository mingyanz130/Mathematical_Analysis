# test_specs.py
"""Python Essentials: Unit Testing.
<Mingyan Zhao>
<Math 345>
<10/09/2018>
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    #test cases for all conditions
    assert specs.smallest_factor(1) == 1, "failed on 1"
    assert specs.smallest_factor(2) == 2, "failed on 2"
    assert specs.smallest_factor(3) == 3, "failed on 3"
    assert specs.smallest_factor(5) == 5, "failed on 5"
    assert specs.smallest_factor(6) == 2, "failed on 6"
    assert specs.smallest_factor(7) == 7, "failed on 7"
    assert specs.smallest_factor(4) == 2, "failed on 4"
    assert specs.smallest_factor(9) == 3, "failed on 9"
    assert specs.smallest_factor(10) == 2, "failed on 10"
    assert specs.smallest_factor(11) == 11, "failed on 11"
    assert specs.smallest_factor(12) == 2, "failed on 12"
     
# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    #test cases for conditions
    assert specs.month_length("September") == 30
    assert specs.month_length("April") == 30
    assert specs.month_length("June") == 30
    assert specs.month_length("November") == 30
    assert specs.month_length("January") == 31
    assert specs.month_length("March") == 31
    assert specs.month_length("May") == 31
    assert specs.month_length("July") == 31
    assert specs.month_length("August") == 31
    assert specs.month_length("October") == 31
    assert specs.month_length("December") == 31
    assert specs.month_length("February") == 28
    assert specs.month_length("February", True) == 29
    assert specs.month_length("Febr", True) == None
    
    
        

# Problem 3: write a unit test for specs.operate().
def test_operate():
    #test cases for all conditions
    assert specs.operate(1, 2, "+") == 3
    assert specs.operate(3, 5, "-") == -2
    assert specs.operate(2, 3, "*") == 6
    assert specs.operate(6, 3, "/") == 2
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4, 0, "/")
    assert excinfo.value.args[0] == "division by zero is undefined"
    
    with pytest.raises(TypeError) as excinfo:
        specs.operate(4, 0, 2)
    assert excinfo.value.args[0] == "oper must be a string"
    
    with pytest.raises(ValueError) as excinfo:
        specs.operate(4, 0, "^")
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    #initiate some values
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    assert frac_n2_3.denom == 3
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction(1.3, 3)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"



def test_fraction_str(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(2, 1)) == "2"

def test_fraction_float(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)

def test_fraction_add(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 + frac_1_2 == specs.Fraction(1, 1)
    assert frac_1_3 + frac_1_3 == specs.Fraction(2, 3)
    assert frac_n2_3 + frac_1_2 == specs.Fraction(1, -6)

def test_fraction_sub(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 - frac_1_3  == specs.Fraction(1, 6)
    assert frac_1_3 - frac_1_3 == 0
    assert frac_n2_3 - frac_1_3 == -1

def test_fraction_mul(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 * frac_1_2 == specs.Fraction(1, 4)
    assert frac_1_3 * frac_1_2== specs.Fraction(1, 6)
    assert frac_n2_3 * frac_1_2 == specs.Fraction(2, -6)

def test_fraction_truediv(set_up_fractions):
    #test cases for all conditions
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 / frac_1_3  == specs.Fraction(3, 2)
    assert frac_1_3 / frac_1_3 == specs.Fraction(1, 1)
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(0, 0) == 0
    assert excinfo.value.args[0] == "denominator cannot be zero"

    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_n2_3 / specs.Fraction(0, -12)  == specs.Fraction(8, -12)
    assert excinfo.value.args[0] == "cannot divide by zero"
    
    
# Problem 5: Write test cases for Set.
def test_count_sets():
    """test count sets if it meets all the conditions
    """
    assert specs.count_sets(["1022","1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0210", "2110", "1020"]) == 6, ("algorithm failed")
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1022","1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0210", "2110"])
    assert excinfo.value.args[0] == "there are not exactly 12 cards"
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["10221","1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0210", "2110", "1020"]) == 11
    assert excinfo.value.args[0] == "one or more cards does not have exactly 4 digits"
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1122","1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0210", "2110", "1020"]) == 11
    assert excinfo.value.args[0] == "the cards are not all unique"
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1022","1123", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0210", "2110", "1020"]) == 11
    assert excinfo.value.args[0] == "one or more cards has a character other than 0, 1, or 2."

def test_is_set():
    # test if is_test function working
    assert specs.is_set("1122", "2112", "0102") == True
    assert specs.is_set("1222", "2112", "0102") == False
    
