import pandas as pd
from pathlib import Path

CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

FN1 = DATA_DIR / '고금계과제1_v3.0_201301-202408.csv'

class FNData:

    NUMERIC_DATA = [
        '종가(원)',
        '수정주가(원)',
        '수정계수',
        '수익률 (1개월)(%)',
        '시가총액 (상장예정주식수 포함)(백만원)',
        '시가총액 (보통-상장예정주식수 포함)(백만원)',
        '기말발행주식수 (보통)(주)',
        '보통주자본금(천원)',
        '자본잉여금(천원)',
        '이익잉여금(천원)',
        '자기주식(천원)',
        '이연법인세부채(천원)',
        '매출액(천원)',
        '매출원가(천원)',
        '이자비용(천원)',
        '영업이익(천원)',
        '총자산(천원)'
        ]


    def __init__(self, filepath=None, encoding='utf-8'):
        if not filepath:
            filepath = FN1
        
        self.fn1_df = FNData._preprocess_dataguide_csv(filepath, encoding=encoding)
        self.items = self.fn1_df['Item Name '].unique()
        self.symbol_to_name = self.fn1_df[['Symbol', 'Symbol Name']].drop_duplicates().set_index('Symbol').to_dict()['Symbol Name']
        self.name_to_symbol = {v:k for k, v in self.symbol_to_name.items()}

        self._preprocess1()

    
    @staticmethod
    def _preprocess_dataguide_csv(
            fn_file_path, 
            cols=['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency',], # 날짜가 아닌 컬럼들
            skiprows=8, 
            encoding="cp949",
            ):
        fn_df = pd.read_csv(fn_file_path, encoding=encoding, skiprows=skiprows, thousands=",")
        fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")

        return fn_df
    
    @staticmethod
    def _get_panel_df(molten_df, item_name):
        panel_df = molten_df.loc[molten_df['Item Name '] == item_name]
        panel_df = panel_df.pivot(index='date', columns='Symbol', values='value')
        panel_df = panel_df.reset_index()
        
        panel_df = panel_df.set_index('date', inplace=False)
        panel_df.sort_index(inplace=True)
        
        return panel_df 
    
    @staticmethod
    def _filter_univ(univ_list, panel_df, is_copy=True):
        if is_copy:
            return panel_df[univ_list].copy()
        else:
            return panel_df[univ_list]

    def _preprocess1(self):
        adj_close_temp = FNData._get_panel_df(self.fn1_df, '수정주가(원)')
        adj_close_temp.dropna(axis=1, how='all', inplace=True)
        
        self.univ_list = adj_close_temp.columns

        sector_df = FNData._filter_univ(self.univ_list, FNData._get_panel_df(self.fn1_df, 'FnGuide Sector') )
        is_under_supervision_df = FNData._filter_univ(self.univ_list, FNData._get_panel_df(self.fn1_df, '관리종목여부') )
        is_trading_halt_df = FNData._filter_univ(self.univ_list, FNData._get_panel_df(self.fn1_df, '거래정지여부') )

        is_under_supervision_mapping = {
            '정상': True,
            '관리': False,
        }
        is_trading_halt_mapping = {
            '정상': True,
            '정지': False,
        }

        is_under_supervision_df = is_under_supervision_df.replace(is_under_supervision_mapping).infer_objects(copy=False)
        is_trading_halt_df = is_trading_halt_df.replace(is_trading_halt_mapping).infer_objects(copy=False)
        
        self.univ_df = ~sector_df.isnull() & (sector_df != '금융') & is_under_supervision_df & is_trading_halt_df
        self.univ_list = self.univ_df.columns
    
    def get_univ_list(self):
        return self.univ_list

    def get_items(self):
        return self.items

    def get_data(self, item_name):
        assert item_name in self.items, f"{item_name} is not in the item list"

        panel_df = FNData._get_panel_df(self.fn1_df, item_name)
        panel_df = FNData._filter_univ(self.univ_list, panel_df)

        if item_name in FNData.NUMERIC_DATA:
            obj_cols = panel_df.select_dtypes('object').columns
            panel_df[obj_cols] = panel_df[obj_cols].replace(',', '', regex=True).infer_objects(copy=False) 
            panel_df[obj_cols] = panel_df[obj_cols].apply(pd.to_numeric, errors='coerce')
        
        if item_name == '수익률 (1개월)(%)':
            panel_df = panel_df / 100
        
        if item_name == '시가총액 (상장예정주식수 포함)(백만원)' or item_name == '시가총액 (보통-상장예정주식수 포함)(백만원)':
            panel_df = panel_df * 100
        
        masked_df = panel_df * self.univ_df

        return masked_df
        

