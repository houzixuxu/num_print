#输入DataFrame，得到不同的格式的sheet
# 并给定初始位置的行数和 列数
#1. 单个DataFrame 铺满表
#2. 多个DataFrame 横向铺满/纵向铺满 附加title
#3. 多个DataFrame 横向铺满 并画图   附加title
#4. 颜色 格式
import pandas as pd
from collections import Counter


class To_Excel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.writer = pd.ExcelWriter(self.file_path)
        pass
    
    def main(self):
        '''
        自定义excel结构，不同的sheet，不同的排布，最后要save，writer.save()
        '''
        #self.
        pass
    
    def save(self):
        self.writer.save()
    #单个df 铺满表
    def one_df_to_excel(self, df_, sheet_name='3.分箱'):
        star_row = 0
        if 'category' in df_.columns:
            star_row = len(Counter(df_lst[0]['category']))+5
        df_.to_excel(self.writer, sheet_name=sheet_name, startrow=star_row)
    
    #多个df，纵向铺满
    def multi_row_to_excel(self, df_lst, sheet_name='2.变量分析'):
        star_row=0
        if 'category' in df_lst[0].columns:
            star_row = len(Counter(df_lst[0]['category']))+5
        star_row = 2
        for i in df_lst:
            i.to_excel(self.writer, sheet_name=sheet_name, startrow=star_row)
            star_row+=4+i.shape[0]
            
    #多个df，横向铺满，并画图
    def multi_col_to_excel(self, df_lst, sheet_name='4.模型效果'):
        star_col=0
        star_row=0
        #if 'category' in df_lst[0].columns:
        #    star_row = len(Counter(df_lst[0]['category']))+5
        #star_row = len(Counter(df_lst[0]['category']))+5
        for i in df_lst:
            i.to_excel(self.writer, sheet_name=sheet_name, startrow=star_row, startcol=star_col)
            star_col+=9+i.shape[1]
