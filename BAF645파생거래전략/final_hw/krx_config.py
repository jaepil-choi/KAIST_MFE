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

ROOM = 10

# Define schema: column_name: (dtype, max_length)
H5_SCHEMA = {
    'ISU_CD': ('object', ROOM + 12),          # Base length 12 + ROOM=10 => 22
    'ISU_SRT_CD': ('object', ROOM + 8),      # Base length 8 + ROOM=10 => 18
    'ISU_NM': ('object', ROOM + 26),         # Base length 26 + ROOM=10 => 36
    'TDD_CLSPRC': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'FLUC_TP_CD': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'CMPPREVDD_PRC': ('object', ROOM + 1),   # Base length 1 + ROOM=10 => 11
    'TDD_OPNPRC': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'TDD_HGPRC': ('object', ROOM + 1),       # Base length 1 + ROOM=10 => 11
    'TDD_LWPRC': ('object', ROOM + 1),       # Base length 1 + ROOM=10 => 11
    'IMP_VOLT': ('object', ROOM + 5),        # Base length 5 + ROOM=10 => 15
    'NXTDD_BAS_PRC': ('object', ROOM + 10),  # Base length 10 + ROOM=10 => 20
    'ACC_TRDVOL': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'ACC_TRDVAL': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'ACC_OPNINT_QTY': ('object', ROOM + 1),  # Base length 1 + ROOM=10 => 11
    'SECUGRP_ID': ('object', ROOM + 2)       # Base length 2 + ROOM=10 => 12
}


H5_SCHEMA = {
    'ISU_CD': ('object', ROOM + 12),          # Base length 12 + ROOM=10 => 22
    'ISU_SRT_CD': ('object', ROOM + 8),      # Base length 8 + ROOM=10 => 18
    'ISU_NM': ('object', ROOM + 26),         # Base length 26 + ROOM=10 => 36
    'TDD_CLSPRC': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'FLUC_TP_CD': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'CMPPREVDD_PRC': ('object', ROOM + 1),   # Base length 1 + ROOM=10 => 11
    'TDD_OPNPRC': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'TDD_HGPRC': ('object', ROOM + 1),       # Base length 1 + ROOM=10 => 11
    'TDD_LWPRC': ('object', ROOM + 1),       # Base length 1 + ROOM=10 => 11
    'IMP_VOLT': ('object', ROOM + 5),        # Base length 5 + ROOM=10 => 15
    'NXTDD_BAS_PRC': ('object', ROOM + 10),  # Base length 10 + ROOM=10 => 20
    'ACC_TRDVOL': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'ACC_TRDVAL': ('object', ROOM + 1),      # Base length 1 + ROOM=10 => 11
    'ACC_OPNINT_QTY': ('object', ROOM + 1),  # Base length 1 + ROOM=10 => 11
    'SECUGRP_ID': ('object', ROOM + 2)       # Base length 2 + ROOM=10 => 12
}
