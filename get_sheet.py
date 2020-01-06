import copy
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

class Get_res_DataFrame:
    '''
    sheet1：数据概况
    sheet2：变量的大小，效果，相关性    ok
    sheet3：分箱结果及woe             ok
    sheet4：按单一类别分  输入 df[['类别', 'final_score']]  cut_line依据  输出 并计算ks
    
    
    通过输入不同的df来返回不同的df分析
    ins，oot，oot2 第一个函数
    新老客区分 第一个函数 输入df_new， df_old， type_train
    月份区分  第一个函数 输入df_new ， df_old， month
    '''
    
    def __init__(self, lr, df, df_bin, df_woe, use_lst, woe_dic, type_train='type_train',  y='is_7_p'):
        self.df = df
        self.df_bin = df_bin
        self.df_woe = df_woe
        self.use_lst = use_lst
        self.woe_dic = woe_dic
        self.type_train = type_train
        self.model = lr
        self.y = y
    
    def main(self):
        
        print('d2_1 = self.get_2_1_imp()',#依次放好,
        'd2_2 = self.get_2_2_des()',
        'd2_3 = self.get_2_3_corr()',
        '''d3 = self.get_bin_ins_oot(type_lst=['ins', 'oot', 'oot2'])''' ) #一整个
        
        #return d2_1, d2_2, d2_3, d3
        
    #df, df_woe, use_lst, cal_iv, type_train,cal_psi ,lr
    def get_2_1_imp(self, df):
        d1 = DataFrame(index=self.use_lst)
        cover_dic = dict(df[use_lst].notnull().sum())
        d1['auc'] = [round(0.5+abs(0.5-roc_auc_score(df[self.y], df[i])), 3) for i in self.use_lst]
        #d1['ks'] = [round(max(abs(roc_curve(df[self.y],df[name])[0]- roc_curve(df[self.y],df[name])[1])), 3) for name in self.use_lst]
        d1['ks'] = [round(float(self.ks_calc_cross(df, name, self.y)[0]['gap']), 3) for name in self.use_lst]
        d1['ins_iv'] = [round(self.cal_iv(df[df[self.type_train]=='ins'], name, self.y), 3) for name in self.use_lst]
        d1['oot_iv'] = [round(self.cal_iv(df[df[self.type_train]=='oot'], name, self.y), 3) for name in self.use_lst]
        
        d1['coef'] = [round(i, 4) for i in self.model.coef_[0]]
        #d1['importance'] = self.model.feature_importances_
        d1 = d1.reset_index()
        d1['psi'] = [round(self.cal_psi(df, name), 5) for name in self.use_lst]
        d1['vif'] = [round(variance_inflation_factor(np.matrix(df[self.use_lst]), i),3) for i in range(len(self.use_lst))]
        #d1['fill_missing_data'] = [fill_na_dic[name] for name in self.use_lst]
        #d2_1 = d1
        d1.index = range(1, d1.shape[0]+1)
        return d1
    
    #df, use_lst, type_train
    def get_2_2_des(self):
        df = self.df[self.df[self.type_train].isin(['ins', 'oot'])]
        df_data_des = df[self.use_lst].describe().T 
        
        
        cover_dic = dict(df[use_lst].notnull().sum())
        
        df_data_des = df_data_des.reset_index()
        df_data_des['cover'] = df_data_des['index'].apply(lambda x: round(cover_dic[x]/df.shape[0], 4))
        df_data_des.index = df_data_des['index']
        df_data_des.drop(columns=['index', 'count'], inplace=True)
        d2_2 = df_data_des.reset_index()
        d2_2.index = range(1, d2_2.shape[0]+1)
        return d2_2
    
    #df_woe, use_lst
    def get_2_3_corr(self):
        corr = np.corrcoef(np.array(self.df_woe[self.use_lst]).T)
        d2_3 = DataFrame(corr, columns=range(len(self.use_lst)), index=self.use_lst).reset_index()
        d2_3.index = range(1, d2_3.shape[0]+1)
        return d2_3
    
    #df_bin, use_lst, #type_lst#, type_train, woe_dic
    def get_bin_ins_oot(self, type_lst=['ins', 'oot', 'oot2']):
        res = []
        for loc, i in enumerate(type_lst):
            lst = []
            df_tmp = self.df_bin[(self.df_bin[self.type_train]==i)]

            for name in self.use_lst:
                #ks_lst = list(self.ks_calc_cross(df_tmp, name, self.y)[1]['gap'])
                #while  len(ks_lst) > df_tmp.shape[0]:
                #    ks_lst.pop()
                #while len(ks_lst) < df_tmp.shape[0]:
                #    ks_lst.append(0)
                #print(ks_lst)
                dd_tmp = df_tmp.groupby(name).sum()[[self.y, 'count']]
                dd_tmp['bad_rate'] = dd_tmp[self.y]/dd_tmp['count']
                dd_tmp = dd_tmp.reset_index()
                dd_tmp['woe'] = dd_tmp[name].apply(lambda x: self.woe_dic[name][x])
                dd_tmp.sort_values(by='bad_rate', inplace=True)        
                dd_tmp['sort_key'] = [float(i.split(',')[0][1:]) if i[0]=='(' else float('inf') for i in dd_tmp[name]]
                #print(dd_tmp)
                dd_tmp.sort_values(by='sort_key', inplace=True)
                dd_tmp.drop(columns=['sort_key'], inplace=True)
                name1 = '-'
                d = DataFrame(columns=['slice', 'bad', 'count', 'bad_rio', 'woe'],
                                  data=[[str(name1), '-', '-', '-','-']]+dd_tmp.values.tolist()[:], 
                                      index=[[name]]+['-']*dd_tmp.shape[0])
                if loc < 1:
                    split_name = '<-->'+str(i)
                else:
                    split_name = str(type_lst[loc-1])+'<-->'+str(i)
                d[split_name] = [split_name for i in range(d.shape[0])]
                d = d[[split_name, 'slice', 'bad', 'count', 'bad_rio', 'woe' ]]                 
                lst.append(d)
            res.append(lst)  
        return pd.concat((pd.concat(i for i in res[i]) for i in range(len(type_lst))),axis=1)
    
    #按照类别做DataFrame
    def get_categories_df(self, df, cate='type_new', base_cut='ins', y='final_score'):
        
        df_tmp = copy.deepcopy(df[[cate, self.y, y]])
        df_tmp.rename(columns={cate:'category', self.y:'bad'}, inplace=True)
        cut_line = list(np.percentile(list(df_tmp[df_tmp['category']==base_cut][y]), range(1, 101,10)))
        #np.percentile出来的是np.array格式
        cut_line[0] = -float('inf')
        cut_line.append(float('inf'))
        df_tmp['bins'] = pd.cut(df_tmp[y], bins=cut_line)
        df_tmp['count'] = [1 for i in range(df_tmp.shape[0])]
        #print(df_tmp)
        
        ks_lst = []
        for i in sorted(Counter(df_tmp['category']).keys()):
            #print(df_tmp[df_tmp['category']==i].shape)
            lst = list(ks_calc_cross(df_tmp[df_tmp['category']==i], 'bins', 'bad')[1]['gap'])
            #print(lst)
            while len(lst) < 10:
                lst = [0]+lst
            ks_lst.extend(lst)
        
        
        df = df_tmp.groupby(['category', 'bins']).sum()[['bad', 'count']]
        df = df.reset_index()
        df['bad_rate'] = df['bad']/df['count']
        df['ks'] = ks_lst
        #print(df)
        for i in ['bad', 'count', 'bad_rate', 'ks']:
            df[i] = df[i].astype(float)
        #df[['bad', 'count', 'bad_rate', 'ks']] = df[['bad', 'count', 'bad_rate', 'ks']].astype(float)
        #df = df.astype(str)
        df[['bad', 'count', 'bad_rate', 'ks'] ]= df[['bad', 'count', 'bad_rate', 'ks']].fillna(0)
        #添加几行用来画画
        #
        #n = len(Counter(df_tmp[cate]))
        #length = df.shape[0]//n
        #for i in range(n):
        #    
        #df[:length]
        #print(df)
        #
        df.index = range(1, df.shape[0]+1)
        return df
    def ks_calc_cross(self,data,pred,y_label):
        '''
        功能: 计算KS值，输出对应分割点和累计分布函数曲线图
        输入值:
        data: 二维数组或dataframe，包括模型得分和真实的标签
        pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
        y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
        输出值:
        'ks': KS值，'crossdens': 好坏客户累积概率分布以及其差值gap
        '''
        crossfreq = pd.crosstab(data[pred],data[y_label])
        crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
        crossdens['gap'] = abs(crossdens[0] - crossdens[1])
        ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
        return ks,crossdens
    
    def cal_iv(self,df1, x, y='is_7_p'):
        df = copy.deepcopy(df1)
        if 'count' not in df.columns:
            df['count'] = [1 for i in range(df.shape[0])]
        df_tmp = df[[x,'count', y]].groupby(x).sum()
        df_tmp['good'] = df_tmp['count'] - df_tmp[y]
        df_tmp[y] = df_tmp[y].apply(lambda x: max(x, 0.00001)/sum(df_tmp[y]))
        df_tmp['good'] = df_tmp['good'].apply(lambda x: max(x, 0.00001)/sum(df_tmp['good']))
        #计算woe
        df_tmp['woe'] = np.log(df_tmp[y]/df_tmp['good'])
        #计算iv
        df_tmp['iv'] = (df_tmp[y]-df_tmp['good']) * df_tmp['woe']
        return df_tmp['iv'].sum()
    
    
    #计算psi
    def cal_psi(self, df_sf_bin, name,  lst=['ins', 'oot']):
        name1, name2 = lst
        
        df_in = copy.deepcopy(df_sf_bin[df_sf_bin['type_train']==name1])
        sum_1 = df_in.shape[0]
        df_in['count1'] = [1 for i in range(sum_1)]
        df_in = df_in.groupby(name).sum()[['count1']]
        
        df_out = copy.deepcopy(df_sf_bin[df_sf_bin['type_train']==name2])
        sum_2 = df_out.shape[0]
        df_out['count2'] = [1 for i in range(sum_2)]
        df_out = df_out.groupby(name).sum()[['count2']]
        df_psi = pd.concat((df_in, df_out), axis=1)
        #计算psi
        df_psi['count1'] = df_psi['count1'].apply(lambda x: x/sum_1)
        df_psi['count2'] = df_psi['count2'].apply(lambda x: x/sum_2)
        #处理出现0的空箱
        df_psi[['count1', 'count2']].replace(0, 0.001, inplace=True)
        #
        df_psi['psi_tmp'] = df_psi['count1']/df_psi['count2']
        df_psi['psi_tmp'] = df_psi['psi_tmp'].apply(lambda x: math.log(x))
       # print(df_psi)
        df_psi['psi'] = (df_psi['count1'] - df_psi['count2'])*df_psi['psi_tmp']
        #df_psi
        return sum(df_psi['psi'])
        
if __name__ == '__main__':
    
    s = '''
           c=Get_res_DataFrame(lr, a.df, a.df_bin, df_pb_woe, use_lst,a.woe_dic, type_train='type_train', y='is_7_p')
           d2_1 = c.get_2_1_imp(df_pb_woe[df_pb_woe['customer_type_old']=='old_customer'])
           d2_2 = c.get_2_2_des()
           d2_3 = c.get_2_3_corr()
           
           d3 = c.get_bin_ins_oot(type_lst=['ins', 'oot'])
           d4 = c.get_categories_df(df_pb_all,cate='type_train',base_cut='ins', y='final_score')
           #
           df_new = df_pb_all[df_pb_all['customer_type_old']=='new_customer']
           df_old = df_pb_all[df_pb_all['customer_type_old']=='old_customer']
           #
           d5_1 = c.get_categories_df(df_new,cate='type_train',base_cut='ins', y='final_score')
           d5_2 = c.get_categories_df(df_old,cate='type_train',base_cut='ins', y='final_score')
           
           d6_1 = c.get_categories_df(df_new,cate='month',base_cut='0', y='final_score')
           d6_2 = c.get_categories_df(df_old,cate='month',base_cut='0', y='final_score')
        '''
    
