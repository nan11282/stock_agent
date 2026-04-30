from tools import exchange_prefix


def test_sh_main_board():
    assert exchange_prefix("600028") == "sh"
    assert exchange_prefix("601398") == "sh"


def test_sh_star_market():
    # 688xxx 科创板也是上海
    assert exchange_prefix("688981") == "sh"


def test_sz_main_and_growth():
    assert exchange_prefix("000001") == "sz"
    assert exchange_prefix("002594") == "sz"
    assert exchange_prefix("300750") == "sz"
