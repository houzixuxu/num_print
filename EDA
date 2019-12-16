#数据预处理，异常值，方差为0（max==min），iv<0.01, auc<0.53, cover..,psi>0.2
#打上ins，oot，oot2标签
class EDA:
    def __init__(self, df, use_lst, y='is_7_p'):
        self.df = df
        self.use_lst = use_lst
        self.y = y
        pass   
    
    def main(self):
        '''
        '''
        lst = copy.deepcopy(self.use_lst)
        #根据覆盖率筛选，并删除
        self.drop_lst_cover = self.cal_cover(self.df[self.df['customer_type_old']=='old_customer'], lst, 0.01)
        lst = list(set(lst)-set(self.drop_lst_cover))
        #根据std==0，并删除
        self.drop_lst_std = self.cal_std(self.df, lst)
        lst = list(set(lst)-set(self.drop_lst_std))
        #根据auc<52，并删除        
        self.drop_lst_auc = self.cal_auc(self.df,lst, 53)
        lst = list(set(lst)-set(self.drop_lst_auc))
        return lst
        
    def cal_cover(self, df, lst, num):
        d = dict(df[lst].notnull().sum()/df.shape[0])
        res = []
        for i in d.items():
            if i[1] < num:
                res.append(i[0])
        return res
    
    def cal_std(self, df, lst):
        res = []
        for i in zip(lst, [max(df[i])-min(df[i]) for i in lst]):
            if i[1] == 0:
                res.append(i[0])
        return res
    
    def cal_auc(self, df, lst, num):
        res1 = []
        res = []
        for i in lst:
            #print(i)
            df_tmp = df[df[i].notnull()]
            #print(df_tmp[[y, i]])
            if len(Counter(df_tmp[y]))==1 or df_tmp.shape[0]==0:
                res1.append([i, 0])
                continue
            else:
                res1.append([i, abs(0.5-roc_auc_score(df_tmp[self.y],df_tmp[i]))+0.5])
        for i in res1:
            if i[1] < (num/100):
                res.append(i[0])
        return res
    
    
    def cal_num_psi(self):
        pass
    def cal_psi(self):
        pass
    def cal_iv(self):
        pass
    def cal_ks(self):
        pass
    def get_woe_lst(self):
        pass
    
def cal_num_psi(df_sf, name, test_name='type_new', lst=['ins', 'oot']):    
    pass

    
#计算psi
def cal_psi(df_sf_bin,name,  lst=['ins', 'oot']):
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

#计算 KS值
def ks_calc(pre,label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'crossdens': 好坏客户累积概率分布以及其差值gap
    '''
    crossfreq = pd.crosstab(pre,label)
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks,crossdens

#计算 KS值
def ks_calc_cross(data,pred,y_label):
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

def ks_calc_auc(data,pred,y_label):

    fpr,tpr,thresholds= roc_curve(data[y_label],data[pred])
    ks = max(tpr-fpr)
    return ks

def cal_iv(df1, x, y='is_7_p'):
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

def get_woe_lst(df1, x, y='is_7_p'):
    df = copy.deepcopy(df1)
    if 'count' not in df.columns:
        df['count'] = [1 for i in range(df.shape[0])]
    df_tmp = df[[x,'count', y]].groupby(x).sum()
    df_tmp['good'] = df_tmp['count'] - df_tmp[y]
    df_tmp[y] = df_tmp[y].apply(lambda x: max(x, 0.00001)/sum(df_tmp[y]))
    df_tmp['good'] = df_tmp['good'].apply(lambda x: max(x, 0.00001)/sum(df_tmp['good']))
    #计算woe
    df_tmp['woe'] = np.log(df_tmp[y]/df_tmp['good'])
    #df_tmp['woe'] = pow(np.log(df_tmp[y]/df_tmp['good']), 2)
    #df_tmp['woe'] = pow(df_tmp[y]/df_tmp['good'], )
    ##计算iv
    #df_tmp['iv'] = (df_tmp[y]-df_tmp['good']) * df_tmp['woe']
   # woe_dic = dict(zip(list(df_tmp.index), list(df_tmp['woe'])))
    return list(df_tmp['woe'])

def get_result(model,x_test, y_test):
    y_pre_test = DataFrame({'a':[i[1] for i in model.predict_proba(x_test)]}, index=y_test.index)['a']
    result_dic = {}
    auc = roc_auc_score(y_test, y_pre_test)
    ks = ks_calc(y_pre_test, y_test)[0]['gap'].values
    result_dic['auc'] = auc
    result_dic['ks'] = ks
    return  result_dic
    
def get_score_lst(xgb):
    use_xgb_lst_1 = [i for i in zip(list(x_train.columns), xgb.feature_importances_) if i[1] >0]
    use_xgb_name_lst_1 = [i[0] for i in use_xgb_lst_1]
    return use_xgb_lst_1, use_xgb_name_lst_1
def cal_person(data, x, y):
    d_tmp = data[[x, y]]
    d_tmp = d_tmp[~(d_tmp[x].isnull()) & ~(d_tmp[y].isnull())]
    a, b = d_tmp[x], d_tmp[y]
    return pearsonr(a, b)
def get_num_str_lst(df):
    str_lst = []
    num_lst = []
    for i in df.dtypes.items():
        if str(i[1]) == 'object':
            str_lst.append(i[0])
        else:
            num_lst.append(i[0])
    #print(len(num_lst), len(str_lst))
    return num_lst, str_lst
