import pandas as pd
from pathlib import Path

CWD = Path('.').resolve()
DATA_DIR = CWD / 'data'

class FnData:

    NUMERIC_DATA = [
        '종가(원)',
        '수정주가(원)',
        '수정계수',
        '수익률 (1개월)(%)',
        # '상장주식수 (보통)(주)',
        # '시가총액 (상장예정주식수 포함)(백만원)',
        # '시가총액 (보통-상장예정주식수 포함)(백만원)',
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

    UNIV_REFERENCE_ITEMS = [
        # '수정주가(원)', # 2624
        # '종가(원)', # 2624
        '수익률 (1개월)(%)', # 2616
        ]
    
    DIV_BY_100 = [
        '수익률 (%)',
        '수익률 (1개월)(%)',
        ]

    MULTIPLY_BY_1000 = [
        '보통주자본금(천원)',
        '자본잉여금(천원)',
        '이익잉여금(천원)',
        '자기주식(천원)',
        '이연법인세부채(천원)',
        '매출액(천원)',
        '매출원가(천원)',
        '이자비용(천원)',
        '영업이익(천원)',
        '총자산(천원)',
        ]

    FN_INDEX_COLS = ['date', 'Symbol', 'Symbol Name',]

    def __init__(self, filepath, encoding='utf-8'):
        if not filepath:
            raise ValueError("파일 경로를 입력해 주세요 예: ./data/고금계과제1.csv")
        
        self.fn1_df = FnData._melt_dataguide_csv(filepath, encoding=encoding)
        self.items = self.fn1_df['Item Name '].unique()
        self._symbol_to_name = self.fn1_df[['Symbol', 'Symbol Name']].drop_duplicates().set_index('Symbol').to_dict()['Symbol Name']
        self._name_to_symbol = {v:k for k, v in self._symbol_to_name.items()}

        self.long_format_df = self._pivot_numerics()
        self._preprocess_numerics()

        self.filter_dfs = self._make_filters()
        self._apply_filters()

        self.univ_list = self._get_univ_list()

    
    @staticmethod
    def _melt_dataguide_csv(
            fn_file_path, 
            cols=['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency',], # 날짜가 아닌 컬럼들
            skiprows=8, 
            encoding="cp949",
            ):
        fn_df = pd.read_csv(fn_file_path, encoding=encoding, skiprows=skiprows, thousands=",")
        fn_df = fn_df.melt(id_vars=cols, var_name="date", value_name="value")

        return fn_df
    
    def _pivot_nonnumeric(self, item_name):
        # string value를 가진 FnGuide Sector의 경우 pivot_table이 안됨. 
        nonnumeric_data = self.fn1_df[self.fn1_df['Item Name '] == item_name].pivot(
            index=FnData.FN_INDEX_COLS,
            columns='Item Name ',
            values='value',
        ).reset_index()

        return nonnumeric_data

    def _pivot_numerics(self):
        # numeric data를 가진 경우 pivot_table이 가능. non-numeric은 알아서 빠짐.
        numeric_data = self.fn1_df.pivot_table(
            index=FnData.FN_INDEX_COLS,
            columns='Item Name ',
            values='value',
            aggfunc='first',
            dropna=True, # False로 하면 memory error남
        ).reset_index()

        return numeric_data

    def _preprocess_numerics(self):

        obj_cols = self.long_format_df.select_dtypes(include='object').columns
        obj_cols = [obj_col for obj_col in obj_cols if obj_col in FnData.NUMERIC_DATA]
        self.long_format_df[obj_cols] = self.long_format_df[obj_cols].replace(',', '', regex=True).infer_objects(copy=False)
        self.long_format_df[obj_cols] = self.long_format_df[obj_cols].apply(pd.to_numeric, errors='raise') 
        
        return

    def _make_filters(self):
        # Filters
        finance_sector = self._pivot_nonnumeric('FnGuide Sector')
        finance_sector = finance_sector[finance_sector['FnGuide Sector'] == '금융']

        is_under_supervision = self._pivot_nonnumeric('관리종목여부')
        is_under_supervision = is_under_supervision[is_under_supervision['관리종목여부'] == '관리']

        is_trading_halted = self._pivot_nonnumeric('거래정지여부') 
        is_trading_halted = is_trading_halted[is_trading_halted['거래정지여부'] == '정지']

        return [
            finance_sector,
            is_under_supervision,
            is_trading_halted,
        ]

    def _apply_filters(self):
        # left가 사용할 long-format df, right가 filter df
        # date-symbol을 key로 join
        # left join하여 매칭되는 것을 제거

        for filter_df in self.filter_dfs:
            filter_df['_flag_right'] = 1
            self.long_format_df = self.long_format_df.merge(
                filter_df,
                on=['date', 'Symbol',],
                how='left',
                suffixes=('', '_right')
            )

            self.long_format_df = self.long_format_df[ self.long_format_df['_flag_right'].isnull() ] 
            self.long_format_df.drop(columns=[c for c in self.long_format_df.columns if c.endswith('_right')], inplace=True)
            self.long_format_df.reset_index(drop=True, inplace=True)

        return 

    def _get_univ_list(self, reference_item='수익률 (1개월)(%)'):
        assert reference_item in FnData.UNIV_REFERENCE_ITEMS, f"유니버스 구축을 위해 {FnData.UNIV_REFERENCE_ITEMS} 중 하나가 필요합니다." 
        only_existing = self.long_format_df.groupby('Symbol').filter(
            lambda x: x[reference_item].notnull().any()
        )

        return only_existing['Symbol'].unique()
        
    
    def _get_wide_format_df(self, item_name):
        return self.long_format_df.pivot_table(
            index='date',
            columns='Symbol',
            values=item_name,
        )
    
    def get_universe(self):
        return self.univ_list

    def get_items(self):
        return self.items

    def get_data(self, item: list | str | None =None, multiindex: bool =True):

        if isinstance(item, str):
            assert item in self.items, f"{item} is not in the item list"
            assert item in FnData.NUMERIC_DATA, f"{item} is not a numeric data"

            data = self._get_wide_format_df(item)
            data = data.reindex(columns=self.univ_list)
            
            if item in FnData.DIV_BY_100:
                data = data / 100
            elif item in FnData.MULTIPLY_BY_1000:
                data = data * 1000

        elif isinstance(item, list):
            for i in item:
                assert i in self.items, f"{i} is not in the item list"
                assert i in FnData.NUMERIC_DATA, f"{i} is not a numeric data"
            
            data = self.long_format_df.loc[:, FnData.FN_INDEX_COLS + item]
            
            for col in data.columns:
                if col in FnData.DIV_BY_100:
                    data[col] = data[col] / 100
                elif col in FnData.MULTIPLY_BY_1000:
                    data[col] = data[col] * 1000
            
            if multiindex:
                data.drop(columns=['Symbol Name',], inplace=True)
                data.index.name = None
                data.set_index(['date', 'Symbol'], inplace=True)
            
            data = data.reindex(self.univ_list, level=1)
                
        
        elif item is None:
            data = self.long_format_df.copy()
            
            if multiindex:
                data.drop(columns=['Symbol Name',], inplace=True)
                data.index.name = None
                data.set_index(['date', 'Symbol'], inplace=True) 
            
            data = data.reindex(self.univ_list, level=1)
        
        else:
            raise ValueError("""
                             item은 

                             - str (1개 item만 wide-format 반환) 
                             - list (선택한 item들 long-format 반환)
                             - None (전체 long-format 반환)

                             중 하나여야 합니다.
                             (numeric data만 선택 가능)
                             """)
        
        return data

    def symbol_to_name(self, symbol_code):
        return self._symbol_to_name[symbol_code]
    
    def name_to_symbol(self, symbol_name):
        return self._name_to_symbol[symbol_name]
        

