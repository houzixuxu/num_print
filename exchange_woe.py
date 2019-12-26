class Fill_WOE_exchange:
    '''
    df：        输入为DateFrame格式
    use_lst：   应用的数值型字段列表
    type_train：根据某种规则得到的 ins, oot, oot2, oot3
    y：         指定的label
    
    运行main函数，获得常规用法
    '''
    def __init__(self, df, use_lst, type_train, y):
        self.df = df
        self.df['count'] = [1 for i in range(self.df.shape[0])]
        self.use_lst = use_lst
        self.type_train = type_train
        self.y = y
        #self.df_fill = self.fill_missing_data()
        
    def main(self):
        #s = '''
        
        #获得需要填充的字段列表
        self.fill_lst = self.get_small_cover_lst(0.05)
        
        #对部分字段进行填充
        self.df_fill, self.fill_na_dic = self.fill_missing_data(self.fill_lst)
        
        #xgboost获取分箱规则
        self.use_cut_lst = self.get_tree_cut_line(self.df_fill)
        
        #用分箱规则分箱，得到self.df_bin
        self.df_bin = self.get_bin_label(self.df_fill, self.use_cut_lst)
        
        #获得每一个分箱的woe值，并存成字典woe_dic
        self.woe_dic = self.get_woe_dic(self.df_bin[self.df_bin[self.type_train]=='ins'])
        
        #用woe_dic，获得woe转换后的DataFrame
        self.df_woe = self.get_woe_bin(self.df_bin, self.woe_dic)
        
        #'''
        #return s
    
    
    def get_small_cover_lst(self, cover_rate=0.05):
        '''
        小于cover_rate的字段，需要填充处理
        '''
        fill_lst = []
        for i in dict(self.df[self.df[self.type_train]=='ins'][self.use_lst].isnull().sum()/self.df[self.df[self.type_train]=='ins'][self.use_lst].shape[0]).items():
            if i[1] <= cover_rate:
                fill_lst.append(i[0])
        return fill_lst
    
    def fill_missing_data(self, fill_lst):
        '''
        填充缺失值
        '''
        df = copy.deepcopy(self.df)
        df1 = copy.deepcopy(df[df[self.type_train]=='ins'])
        fill_na_dic = {}
        for name in fill_lst:
            #print(name)
            if df[name].isnull().sum() == 0:
                fill_na_dic[name] = 0
            else:
                m = 0.5
                fill_lst = [df1[name].mean(), df1[name].min(),df1[name].min()-1,df1[name].max()+1, df1[name].max(), 0,1]+list(set(np.percentile(df1[name].dropna(), range(1, 101, 10))))
                for i in fill_lst:
                    n = abs(0.5-roc_auc_score(df1[self.y], df1[name].fillna(i)))+0.5
                    if n > m:
                        num = i
                        m = n
                fill_na_dic[name] = num
                df[name] = df[name].fillna(num)
        return df, fill_na_dic

 #决策树分箱  的每个变量的切分点
    def get_tree_cut_line(self, df):
        '''
        #决策树分箱的每个变量的切分点
        '''
        print(len(self.use_lst))
        use_cut_lst = {}
        left_num = -float(np.inf)
        right_num = float(np.inf)
        for name in self.use_lst:
            x1 = df[(df[name].notnull())&(df[self.type_train]=='ins')][[name]]
            y1 = df[(df[name].notnull())&(df[self.type_train]=='ins')][self.y]
            x2 = df[(df[name].notnull())&(df[self.type_train]=='oot')][[name]]
            y2 = df[(df[name].notnull())&(df[self.type_train]=='oot')][self.y]
            min_num = x1.shape[0]/200
            print(min_num)
            print(name, x1.shape, x2.shape)
            if len(Counter(x1[name])) <= 3:
                cut_line = sorted([i-0.1 for i in Counter(x1[name]).keys()])
                cut_line[0] = left_num
                cut_line.append(right_num)
    
            else:
                r_1 = abs(0.5-roc_auc_score(y1, x1))+0.5 
                #55以下 深度1
                #68以下 深度2
                #75以下 深度3
                
                #样本比例权重
                sample_wei=len(y1)//sum(y1)+2
                print(sample_wei)
                
                
                if r_1 < .55:
                    depth = 1
                    ga = 11
                    min_wei = min_num
                elif r_1 < .65:
                    depth = 2
                    ga = 110
                    min_wei = min_num*1.3
                else:
                    depth = 3
                    ga = 100
                    min_wei = min_num*1.2

                xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                       colsample_bynode=1, colsample_bytree=1, gamma=ga, learning_rate=0.1,
                       max_delta_step=0, max_depth=depth, min_child_weight=min_wei, missing=np.nan,
                       n_estimators=1, n_jobs=-1, nthread=None,
                       objective='binary:logistic', random_state=0, reg_alpha=0,
                       reg_lambda=0, scale_pos_weight=sample_wei, seed=None, silent=None,
                       subsample=1, verbosity=0)
                xgb.fit(x1, y1, eval_metric='auc', eval_set=[(x1, y1), (x2, y2)])
                
                cut_line = sorted(xgb.get_booster().trees_to_dataframe()['Split'].dropna())
                cut_line = [left_num] + cut_line + [right_num]
                cut_line = sorted(set(cut_line))
            use_cut_lst[name] = cut_line
            print(len(cut_line)-1)
        return use_cut_lst

    def get_bin_label(self, df_fill,use_cut_lst):
        '''
        用分箱规则，对df_fill分箱
        '''
        df = copy.deepcopy(df_fill)
        for name in self.use_lst:
            df[name] = pd.cut(df[name], bins=use_cut_lst[name])
            df[name] = df[name].astype(str)
           # df1[name] = pd.cut(df[name], bins=use_cut_lst[name])
            d_tmp = df[df[self.type_train]=='ins'].groupby(name).sum()[[self.y, 'count']].reset_index()
            d_tmp['bad_rate'] = d_tmp[self.y]/d_tmp['count']
            d_tmp.sort_values(by='bad_rate', inplace=True)
            print(d_tmp)
        return df
        
     #得到woe字典
    def get_woe_dic(self,df_bin):
        '''
        #根据分箱的结果，计算得到woe字典
        '''
        woe_dic = {}
        df = copy.deepcopy(df_bin[df_bin[self.type_train]=='ins'])
        for name in self.use_lst:
            woe_lst = self.get_woe_lst(df, name, self.y)
            #woe_lst = [i*2 for i in woe_lst]
            print(name, woe_lst)
            woe_dic[name] = dict(zip(sorted(dict(Counter(df[name])).keys()), woe_lst))
        return woe_dic
    #得到woe分箱
    def get_woe_bin(self, df1, woe_dic):
        '''
        #将分箱结果进行woe转换
        '''
        df = copy.deepcopy(df1)
        for name in self.use_lst:
            #print(name)
            df[name] = df[name].apply(lambda x: woe_dic[name][x])
            df[name] = df[name].astype(float)
        return df   
    
    def get_woe_lst(self,df1, x, y='is_7_p'):
        '''
        计算woe_lst
        '''
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
   #     woe_dic = dict(zip(list(df_tmp.index), list(df_tmp['woe'])))
        return list(df_tmp['woe'])
