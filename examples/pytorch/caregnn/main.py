import argparse

import dgl

import torch as th
import torch.optim as optim
from model import CAREGNN
from sklearn.metrics import recall_score, roc_auc_score
from torch.nn.functional import softmax
from utils import EarlyStopping


def main(args):
    # Step 1: 准备数据集; 得到train、val、test的索引; 以及train中正样本的索引; ============================= #
    # 下载数据集
    # dataset = dgl.data.FraudDataset(args.dataset, train_size=0.4)
    # graph = dataset[0]
    # num_classes = dataset.num_classes
    if (args.dataset != "S-FFSD"):
        dataset = dgl.data.FraudDataset(args.dataset, train_size=0.4)
        graph = dataset[0]
        num_classes = dataset.num_classes
        # print("num_classes",num_classes)
    if args.dataset == "S-FFSD":
        from my_add import load_gtan_data
        graph = load_gtan_data(args.dataset, train_size=0.4)
        num_classes = 2

    # 检查cuda是否可用
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # 得到标签
    labels = graph.ndata["label"].to(device)

    # 得到节点特征
    feat = graph.ndata["feature"].to(device)

    # 分布得到train/val/test 索引
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze(1).to(device) # 选出train_mask中不为0的索引，并压缩成一维张量
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    # 得到训练集中正样本的索引
    rl_idx = th.nonzero(
        train_mask.to(device) & labels.bool(), as_tuple=False
    ).squeeze(1)

    graph = graph.to(device)

    # Step 2: 创建模型 =================================================================== #
    model = CAREGNN(
        in_dim=feat.shape[-1],# 输入特征的维度
        num_classes=num_classes,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        activation=th.tanh,
        step_size=args.step_size,
        edges=graph.canonical_etypes,# 将图中的边全变成3元组格式(源节点，边，目标节点)
    )

    model = model.to(device)

    # Step 3: 创建训练所需要的（损失函数，优化器，早停） ===================================================== #
    mask = labels != 2
    filtered_labels = labels[mask]

    # 计算去掉标签为2后的唯一标签和每个标签的数量
    _, cnt = th.unique(filtered_labels, return_counts=True)
    # _, cnt = th.unique(labels, return_counts=True) # 标签中的不重复的元素，并返回每个标签对应的个数（对于二分类问题，个数就是2，也就是0和1）cnt={Tensor:(2,)}
    loss_fn = th.nn.CrossEntropyLoss(weight=1 / cnt)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        # Training and validation using a full graph
        model.train()
        logits_gnn, logits_sim = model(graph, feat)

        # compute loss
        tr_loss = loss_fn(
            logits_gnn[train_idx], labels[train_idx]
        ) + args.sim_weight * loss_fn(logits_sim[train_idx], labels[train_idx])

        tr_recall = recall_score(
            labels[train_idx].cpu(),
            logits_gnn.data[train_idx].argmax(dim=1).cpu(),
        )
        tr_auc = roc_auc_score(
            labels[train_idx].cpu(),
            softmax(logits_gnn, dim=1).data[train_idx][:, 1].cpu(),
        )

        # validation
        val_loss = loss_fn(
            logits_gnn[val_idx], labels[val_idx]
        ) + args.sim_weight * loss_fn(logits_sim[val_idx], labels[val_idx])
        val_recall = recall_score(
            labels[val_idx].cpu(), logits_gnn.data[val_idx].argmax(dim=1).cpu()
        )
        val_auc = roc_auc_score(
            labels[val_idx].cpu(),
            softmax(logits_gnn, dim=1).data[val_idx][:, 1].cpu(),
        )

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # Print out performance
        print(
            "Epoch {}, Train: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f} | Val: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f}".format(
                epoch,
                tr_recall,
                tr_auc,
                tr_loss.item(),
                val_recall,
                val_auc,
                val_loss.item(),
            )
        )

        # Adjust p value with reinforcement learning module
        model.RLModule(graph, epoch, rl_idx)

        if args.early_stop:
            if stopper.step(val_auc, model):
                break

    # Test after all epoch
    model.eval()
    if args.early_stop:
        model.load_state_dict(th.load("es_checkpoint.pt"))

    # forward
    logits_gnn, logits_sim = model.forward(graph, feat)

    # compute loss
    test_loss = loss_fn(
        logits_gnn[test_idx], labels[test_idx]
    ) + args.sim_weight * loss_fn(logits_sim[test_idx], labels[test_idx])
    test_recall = recall_score(
        labels[test_idx].cpu(), logits_gnn[test_idx].argmax(dim=1).cpu()
    )
    test_auc = roc_auc_score(
        labels[test_idx].cpu(),
        softmax(logits_gnn, dim=1).data[test_idx][:, 1].cpu(),
    )

    print(
        "Test Recall: {:.4f} AUC: {:.4f} Loss: {:.4f}".format(
            test_recall, test_auc, test_loss.item()
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN-based Anti-Spam Model")
    parser.add_argument(
        "--dataset",
        type=str,
        # default="yelp",
        default="S-FFSD",
        help="DGL dataset for this model (yelp, or amazon)",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--hid_dim", type=int, default=64, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of layers"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        # default=200,
        default=10,
        help="The max number of epochs. Default: 30",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate. Default: 0.01"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay. Default: 0.001",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.02,
        help="RL action step size (lambda 2). Default: 0.02",
    )
    parser.add_argument(
        "--sim_weight",
        type=float,
        default=2,
        help="Similarity loss weight (lambda 1). Default: 2",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop",
    )

    args = parser.parse_args()
    print(args)
    th.manual_seed(717)
    main(args)
