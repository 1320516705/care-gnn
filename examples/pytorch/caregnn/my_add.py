import dgl
import torch
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split






def load_gtan_data(dataset: str, train_size: float):
    prefix = os.path.join(os.path.dirname(__file__), "data/")
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        # 1、读数据
        data = pd.read_csv(os.path.join(prefix, 'S-FFSD.csv'))
        data = featmap_gen(data.reset_index(drop=True))
        data.replace(np.nan, 0, inplace=True)
        data.to_csv(os.path.join(prefix, 'S-FFSDneofull.csv'), index=None)
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["Labels"] < 2]
        data = data.reset_index(drop=True)

        # 2、构建graph
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        # 3、将【类别特征】做编码映射处理，也是变相的更新data
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        # 4、生成掩码
        num_nodes = g.num_nodes()
        train_idx, test_idx = train_test_split(np.arange(num_nodes), train_size=train_size, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, train_size=0.5, random_state=42)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        # 5、设置节点特征和标签
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feature'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        # 6、保存图
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        from dgl.data.utils import load_graphs
        graph_list, _ = load_graphs(prefix + "graph-S-FFSD.bin")
        g = graph_list[0]
        return g

def featmap_gen(tmp_df=None):
    """
    Handle S-FFSD dataset and do some feature engineering
    :param tmp_df: the feature of input dataset
    """
    # time_span = [2, 5, 12, 20, 60, 120, 300, 600, 1500, 3600, 10800, 32400, 64800, 129600,
    #              259200]  # Increase in the number of time windows to increase the characteristics.
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]  # 设置时间窗口
    time_name = [str(i) for i in time_span]  # 时间窗口名称
    time_list = tmp_df['Time']  # 时间列
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):  # 遍历每一行，trans_idx是索引，trans_feat是当前行特征
        new_df = pd.Series(trans_feat)  # 将特征转换为Series类型，是当前行特征
        temp_time = new_df.Time  # 将当前行的时间列赋值给temp_time
        temp_amt = new_df.Amount  # 将当前行的金额列赋值给temp_amt
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)  # 对当前行而言，找出当前行时间之前的所有行，并且找出当前行时间之后的所有行，是list
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]  # 找出对于当前行的时间而言，在当前时间窗口内的行，直到遍历完所有的时间窗口。
            new_df['trans_at_avg_{}'.format(  # 所有满足条件的行的金额列的平均值、标准差、总和、偏差，行数量，目标数量，位置数量，类型数量
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)