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
