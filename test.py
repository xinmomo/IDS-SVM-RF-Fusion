# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    auc,
)
import warnings
warnings.filterwarnings('ignore')

# 图像显示配置：False表示仅保存不弹窗
SHOW_FIGURES = False
# 全局固定参数配置
RANDOM_STATE = 42
TEST_SIZE = 0.2
# 数据集文件路径
TRAIN_PATH = "./data/KDDTrain+.txt"
TEST_PATH = "./data/KDDTest+.txt"
# 模型与实验核心参数
NORMAL_LABEL = None
CAT_FEATURES = ["protocol_type", "service", "flag"]
K_SELECT = 25
SMOTE_K_NEIGHBORS = 3
REPEAT_RUNS = 5
FUSION_MODE = "weighted"
FUSION_THRESHOLD = 0.3
SVM_C = 5.0
SVM_GAMMA = 0.01
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_LEAF = 2

# NSL-KDD数据集41个特征列名定义
feature_columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# 加载NSL-KDD数据集文本文件
print("="*50)
print("1. 加载NSL-KDD数据集...")
def load_nsl_kdd_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=",", low_memory=False)
    n_cols = df.shape[1]
    if n_cols == 43:
        df.columns = feature_columns[:-1] + ["label", "difficulty"]
    elif n_cols == 42:
        df.columns = feature_columns
    else:
        raise ValueError(f"数据集列数异常：{path} 实际 {n_cols} 列；期望 42 或 43 列。请检查文件格式。")
    df["protocol_type"] = df["protocol_type"].astype(str)
    df["service"] = df["service"].astype(str)
    df["flag"] = df["flag"].astype(str)
    df["label"] = df["label"].astype(str)
    if "difficulty" in df.columns:
        df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce").fillna(0).astype(int)
    return df

train_data = load_nsl_kdd_txt(TRAIN_PATH)
test_data = load_nsl_kdd_txt(TEST_PATH)
print(f"训练集规模：{train_data.shape[0]} 条样本，{train_data.shape[1]} 个特征")
print(f"测试集规模：{test_data.shape[0]} 条样本，{test_data.shape[1]} 个特征")

# 标签处理：提取唯一标签并自动识别正常类标签
train_unique_labels = sorted(train_data["label"].unique())
test_unique_labels = sorted(test_data["label"].unique())
print(f"\n训练集唯一标签数：{len(train_unique_labels)}；示例：{train_unique_labels[:10]}")
print(f"测试集唯一标签数：{len(test_unique_labels)}；示例：{test_unique_labels[:10]}")
label_counts = train_data["label"].value_counts()

if NORMAL_LABEL is None:
    if "normal" in set(train_unique_labels):
        NORMAL_LABEL = "normal"
    else:
        NORMAL_LABEL = str(label_counts.idxmax())
    print(f"\n[提示] 自动识别 NORMAL_LABEL = '{NORMAL_LABEL}'（训练集中频次最高的标签）")
else:
    NORMAL_LABEL = str(NORMAL_LABEL)
    if NORMAL_LABEL not in set(train_unique_labels):
        print(f"\n[警告] NORMAL_LABEL='{NORMAL_LABEL}' 不在训练集标签中，将导致二分类错误并可能训练失败。")
        exit()

# 转换为二分类标签：0为正常，1为攻击
train_data["label_binary"] = (train_data["label"] != NORMAL_LABEL).astype(int)
test_data["label_binary"] = (test_data["label"] != NORMAL_LABEL).astype(int)

# 打印标签分布统计信息
def print_label_stats(df: pd.DataFrame, name: str) -> None:
    normal_count = int((df["label_binary"] == 0).sum())
    attack_count = int((df["label_binary"] == 1).sum())
    total = len(df)
    print(f"\n{name} label distribution (binary):")
    print(f"- Normal (0): {normal_count} ({normal_count/total:.2%})")
    print(f"- Attack (1): {attack_count} ({attack_count/total:.2%})")
    print(f"- 不平衡比：{attack_count/normal_count:.2f} (攻击/正常)")

print_label_stats(train_data, "训练集")
print_label_stats(test_data, "测试集")

# 绘制并保存标签分布可视化图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x="label_binary", data=train_data)
plt.title("Train set label distribution (binary)")
plt.xlabel("Label (0=Normal, 1=Attack)")
plt.ylabel("Count")
plt.xticks([0, 1], ["Normal", "Attack"])
plt.subplot(1, 2, 2)
sns.countplot(x="label_binary", data=test_data)
plt.title("Test set label distribution (binary)")
plt.xlabel("Label (0=Normal, 1=Attack)")
plt.ylabel("Count")
plt.xticks([0, 1], ["Normal", "Attack"])
plt.tight_layout()
plt.savefig("标签分布.png", dpi=300, bbox_inches="tight")
if SHOW_FIGURES:
    plt.show()
else:
    plt.close()
print("已生成并保存：标签分布.png")

# 数据预处理流程
print("\n" + "="*50)
print("2. 数据预处理...")
drop_cols = ["label", "label_binary"]
if "difficulty" in train_data.columns:
    drop_cols.append("difficulty")
X_train_raw = train_data.drop(drop_cols, axis=1)
y_train = train_data["label_binary"].astype(int).values
X_test_raw = test_data.drop(drop_cols, axis=1)
y_test = test_data["label_binary"].astype(int).values

# 分类特征编码处理
X_train_proc = X_train_raw.copy()
X_test_proc = X_test_raw.copy()
for col in CAT_FEATURES:
    X_train_proc[col] = X_train_proc[col].astype(str).fillna("unknown")
    X_test_proc[col] = X_test_proc[col].astype(str).fillna("unknown")
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_proc[CAT_FEATURES] = oe.fit_transform(X_train_proc[CAT_FEATURES])
X_test_proc[CAT_FEATURES] = oe.transform(X_test_proc[CAT_FEATURES])

# 确保所有特征为数值类型
for col in X_train_proc.columns:
    X_train_proc[col] = pd.to_numeric(X_train_proc[col], errors="coerce").fillna(0)
    X_test_proc[col] = pd.to_numeric(X_test_proc[col], errors="coerce").fillna(0)

# 特征标准化处理
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_proc)
X_test_scaled = scaler.transform(X_test_proc)
print("预处理完成：")
print(f"- 训练特征矩阵形状：{X_train_scaled.shape}")
print(f"- 测试特征矩阵形状：{X_test_scaled.shape}")
print(f"- 训练标签形状：{y_train.shape}，测试标签形状：{y_test.shape}")

# 提取特征名称列表用于后续分析
feature_names = X_train_proc.columns.tolist()

# 模型训练与评估主流程
print("\n" + "="*50)
print("3. 模型训练与对比实验...")

# 绘制ROC和PR曲线函数
def plot_roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray, tag: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title(f"ROC Curve - {tag}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    plt.title(f"Precision-Recall Curve - {tag}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{tag}_ROC_PR.png", dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()
    print(f"已生成并保存：{tag}_ROC_PR.png")

# 模型评估函数，计算多维度指标并保存混淆矩阵
def evaluate_model(y_true, y_pred, model_name, y_score: np.ndarray | None = None, plot_curves: bool = False):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (recall + tnr)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (0)', 'Attack (1)'],
                yticklabels=['Normal (0)', 'Attack (1)'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{model_name}_ConfusionMatrix.png", dpi=300, bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()
    print(f"已生成并保存：{model_name}_ConfusionMatrix.png")
    print(f"[模型评估] {model_name} - FPR：{fpr:.4f}，Recall：{recall:.4f}（误报率/召回率）")

    roc_auc = roc_auc_score(y_true, y_score) if (y_score is not None) else None
    pr_auc = average_precision_score(y_true, y_score) if (y_score is not None) else None
    if plot_curves and (y_score is not None):
        plot_roc_pr_curves(y_true, y_score, tag=model_name)
    
    return {
        "模型": model_name,
        "准确率": round(acc, 4),
        "精确率": round(prec, 4),
        "召回率": round(recall, 4),
        "F1值": round(f1, 4),
        "FPR": round(float(fpr), 4),
        "FNR": round(float(fnr), 4),
        "Specificity(TNR)": round(float(tnr), 4),
        "BalancedAcc": round(float(bal_acc), 4),
        "MCC": round(float(mcc), 4),
        "ROC_AUC": round(float(roc_auc), 4) if roc_auc is not None else None,
        "PR_AUC(AP)": round(float(pr_auc), 4) if pr_auc is not None else None,
    }

# 基于F1值的加权投票函数
def weighted_vote_by_f1(pred_a: np.ndarray, pred_b: np.ndarray, w_a: float, w_b: float) -> np.ndarray:
    score = w_a * pred_a + w_b * pred_b
    threshold = 0.5 * (w_a + w_b)
    return (score >= threshold).astype(int)

# 概率融合函数，结合SVM和RF的预测概率
def probability_fusion(svm_score: np.ndarray, rf_score: np.ndarray, svm_w: float = 0.6, rf_w: float = 0.4):
    fusion_score = svm_w * svm_score + rf_w * rf_score
    fusion_pred = (fusion_score > FUSION_THRESHOLD).astype(int)
    return fusion_pred, fusion_score

# 绘制特征重要性可视化图
def plot_feature_importance(selector, feature_names, tag: str):
    mi_scores = selector.scores_
    mi_df = pd.DataFrame({"feature": feature_names, "mutual_info": mi_scores})
    mi_df = mi_df.sort_values(by="mutual_info", ascending=False).head(K_SELECT)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="mutual_info", y="feature", data=mi_df)
    plt.title(f"Top {K_SELECT} Features by Mutual Information - {tag}")
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(f"{tag}_特征重要性.png", dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()
    print(f"已生成并保存：{tag}_特征重要性.png")
    print(f"[特征选择] 前5个重要特征：{mi_df['feature'].tolist()[:5]}，平均互信息：{mi_df['mutual_info'].mean():.4f}")

# 单次实验执行函数，包含多个模型的训练与评估
def train_and_eval_once(run_seed: int):
    results_local = []
    plot_this_run = (run_seed == seeds[0])
    smote = SMOTE(random_state=run_seed, k_neighbors=SMOTE_K_NEIGHBORS)

    # 基础SVM模型训练与评估
    m0 = svm.SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA,
        class_weight="balanced",
        random_state=run_seed,
        probability=True,
        decision_function_shape='ovr'
    )
    m0.fit(X_train_scaled, y_train)
    m0_pred = m0.predict(X_test_scaled)
    m0_score = m0.predict_proba(X_test_scaled)[:, 1]
    results_local.append(
        evaluate_model(y_test, m0_pred, f"M0_SVM(seed={run_seed})", y_score=m0_score, plot_curves=plot_this_run)
    )

    # SMOTE+SVM模型训练与评估
    X_m1_res, y_m1_res = smote.fit_resample(X_train_scaled, y_train)
    m1 = svm.SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA,
        class_weight="balanced", random_state=run_seed,
        probability=True, decision_function_shape='ovr'
    )
    m1.fit(X_m1_res, y_m1_res)
    m1_pred = m1.predict(X_test_scaled)
    m1_score = m1.predict_proba(X_test_scaled)[:, 1]
    results_local.append(evaluate_model(y_test, m1_pred, f"M1_SMOTE+SVM(seed={run_seed})", y_score=m1_score))

    # 特征选择+SVM模型训练与评估
    selector = SelectKBest(mutual_info_classif, k=K_SELECT)
    X_m2_tr = selector.fit_transform(X_train_scaled, y_train)
    X_m2_te = selector.transform(X_test_scaled)
    if plot_this_run:
        plot_feature_importance(selector, feature_names, "M2_FS")
    m2 = svm.SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA,
        class_weight="balanced", random_state=run_seed,
        probability=True, decision_function_shape='ovr'
    )
    m2.fit(X_m2_tr, y_train)
    m2_pred = m2.predict(X_m2_te)
    m2_score = m2.predict_proba(X_m2_te)[:, 1]
    results_local.append(evaluate_model(y_test, m2_pred, f"M2_FS+SVM(seed={run_seed})", y_score=m2_score))

    # SMOTE+特征选择+SVM模型训练与评估
    selector_m3 = SelectKBest(mutual_info_classif, k=K_SELECT)
    X_m3_tr = selector_m3.fit_transform(X_train_scaled, y_train)
    X_m3_te = selector_m3.transform(X_test_scaled)
    X_m3_res, y_m3_res = smote.fit_resample(X_m3_tr, y_train)
    if plot_this_run:
        plot_feature_importance(selector_m3, feature_names, "M3_SMOTE+FS")
    m3 = svm.SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA,
        class_weight="balanced", random_state=run_seed,
        probability=True, decision_function_shape='ovr'
    )
    m3.fit(X_m3_res, y_m3_res)
    m3_pred = m3.predict(X_m3_te)
    m3_score = m3.predict_proba(X_m3_te)[:, 1]
    results_local.append(
        evaluate_model(y_test, m3_pred, f"M3_SMOTE+FS+SVM(seed={run_seed})", y_score=m3_score, plot_curves=plot_this_run)
    )

    # 融合模型训练与评估
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=run_seed,
        n_jobs=-1
    )
    rf.fit(X_m3_res, y_m3_res)
    svm_m4 = svm.SVC(
        kernel="rbf", C=SVM_C, gamma=SVM_GAMMA,
        class_weight="balanced", random_state=run_seed,
        probability=True, decision_function_shape='ovr'
    )
    svm_m4.fit(X_m3_res, y_m3_res)

    rf_pred = rf.predict(X_m3_te)
    svm_pred = svm_m4.predict(X_m3_te)
    rf_score = rf.predict_proba(X_m3_te)[:, 1]
    svm_score = svm_m4.predict_proba(X_m3_te)[:, 1]

    if FUSION_MODE.lower() == "weighted":
        rf_train_pred = rf.predict(X_m3_res)
        svm_train_pred = svm_m4.predict(X_m3_res)

        w_rf = f1_score(y_m3_res, rf_train_pred)
        w_svm = f1_score(y_m3_res, svm_train_pred)
        
        total_weight = w_rf + w_svm
        w_rf, w_svm = w_rf / total_weight, w_svm / total_weight

        fusion_pred, fusion_score = probability_fusion(
            svm_score,
            rf_score,
            svm_w=w_svm,
            rf_w=w_rf
        )

        results_local.append(
            evaluate_model(
                y_test,
                fusion_pred,
                f"M4_Fusion(WeightedF1,seed={run_seed})",
                y_score=fusion_score,
                plot_curves=plot_this_run
            )
        )
        print(f"[M4融合] seed={run_seed} - SVM权重：{w_svm:.4f}，RF权重：{w_rf:.4f} (归一化后)")

    else:
        fusion_pred = np.where((svm_pred + rf_pred) >= 1, 1, 0)
        fusion_score = np.maximum(svm_score, rf_score)
        results_local.append(
            evaluate_model(
                y_test,
                fusion_pred,
                f"M4_Fusion(HardVote,seed={run_seed})",
                y_score=fusion_score,
                plot_curves=plot_this_run
            )
        )

    return results_local

# 多次重复实验以验证稳定性
all_runs_results = []
seeds = [RANDOM_STATE + i for i in range(REPEAT_RUNS)]
for s in seeds:
    print(f"\n--- 开始重复实验：seed={s} ---")
    all_runs_results.extend(train_and_eval_once(s))

# 结果整理与保存
results_df = pd.DataFrame(all_runs_results)
print("\n" + "="*30 + " 各次实验结果（逐次） " + "="*30)
print(results_df.to_string(index=False))
results_df.to_csv("模型性能对比_逐次.csv", index=False, encoding="utf-8-sig")

# 计算均值和标准差并保存
metric_cols = ["准确率", "精确率", "召回率", "F1值", "FPR", "FNR", "Specificity(TNR)", 
               "BalancedAcc", "MCC", "ROC_AUC", "PR_AUC(AP)"]
results_df["模型组"] = results_df["模型"].str.replace(r"\(seed=\d+\)", "", regex=True)
summary_mean = results_df.groupby("模型组")[metric_cols].mean().round(4)
summary_std = results_df.groupby("模型组")[metric_cols].std(ddof=0).round(4)
summary = summary_mean.add_suffix("_mean").join(summary_std.add_suffix("_std")).reset_index()
print("\n" + "="*30 + " 模型性能对比（均值±标准差） " + "="*30)
print(summary.to_string(index=False))
summary.to_csv("模型性能对比_均值标准差.csv", index=False, encoding="utf-8-sig")

# 绘制核心指标对比图
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
sns.barplot(x="模型组", y="召回率_mean", data=summary)
plt.title("Recall Comparison (mean)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0.6, 0.9)
plt.grid(True, alpha=0.3, axis="y")
plt.subplot(1, 3, 2)
sns.barplot(x="模型组", y="F1值_mean", data=summary)
plt.title("F1-Score Comparison (mean)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0.7, 0.9)
plt.grid(True, alpha=0.3, axis="y")
plt.subplot(1, 3, 3)
sns.barplot(x="模型组", y="MCC_mean", data=summary)
plt.title("MCC Comparison (mean)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0.6, 0.8)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("模型核心指标对比_召回率_F1_MCC.png", dpi=300, bbox_inches='tight')
if SHOW_FIGURES:
    plt.show()
else:
    plt.close()
print("已生成并保存：模型核心指标对比_召回率_F1_MCC.png")

# 输出实验完成信息及生成文件列表
print("\n" + "="*50)
print("所有实验完成！生成的文件：")
files = [
    "1. 标签分布.png - 标签分布+不平衡度可视化",
    "2. *_ConfusionMatrix.png - 各模型混淆矩阵",
    "3. *_ROC_PR.png - 核心模型ROC/PR曲线",
    "4. *_特征重要性.png - 特征选择重要性",
    "5. 模型性能对比_逐次.csv - 每次实验的详细指标",
    "6. 模型性能对比_均值标准差.csv - 均值±标准差汇总",
    "7. 模型核心指标对比_召回率_F1_MCC.png - 核心指标可视化",
]
for f in files:
    print(f)