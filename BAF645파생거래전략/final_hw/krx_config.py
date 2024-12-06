# Constants
API_URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "http://data.krx.co.kr",
    "Referer": "http://data.krx.co.kr/contents/MMC/ISIF/isif/MMCISIF011.cmd",
    "Connection": "keep-alive",
}

PAYLOAD_TEMPLATE = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT12502",
    "prodId02": "KRDRVOPEQU", 
    "prodId": "KRDRVOPEQU",
    "ulyId": "KRDRVOPEQU",
    "rghtTpCd": "T",
    "trdDd": "",  # To be filled with date in YYYYMMDD format
    "idxIndMidclssCd": "01"
}

HEADERS_OPTION_HOME = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "http://data.krx.co.kr",
    "Referer": "http://data.krx.co.kr/contents/MMC/ISIF/isif/MMCISIF011.cmd",
    "Connection": "keep-alive",
}


PAYLOAD_OPTION_HOME = {
    "locale": "ko_KR",
    "trdDd": "20241206",
    "prodId": "KRDRVOPEQU",
    "ulyId": "",
    "trdDdBox1": "20241206",
    "trdDdBox2": "20241206",
    "mktTpCd": "T",
    "rghtTpCd": "T",
    "share": 1,
    "money": 3,
    "csvxls_isNo": False,
    "bld": "/dbms/comm/component/drv_clss11"
}

PAYLOAD_TEMPLATE2 = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT12502",
    "locale": "ko_KR",
    "trdDd": "", # To be filled with date in YYYYMMDD format
    "prodId": "KRDRVOPEQU",
    "ulyId": "", # To be filled with specific underlying ID
    "trdDdBox1": "", # To be filled with date in YYYYMMDD format
    "trdDdBox2": "", # To be filled with date in YYYYMMDD format
    "mktTpCd": "T",
    "rghtTpCd": "T",
    "share": 1,
    "money": 3,
    "csvxls_isNo": False,
}