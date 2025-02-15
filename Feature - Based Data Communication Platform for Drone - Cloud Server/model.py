import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import os
from tqdm import tqdm  # 进度条

current_path = os.getcwd()

# ------------------------------------------------------------------------------------
# 1. 分类器模型 (ResNet18 预训练) -- 用于在边缘侧完成初步分类，并得到“分类置信度”
# ------------------------------------------------------------------------------------
class ResNetCIFAR10(nn.Module):
    """
    以预训练的 ResNet18 作为骨干，并修改最后一层输出 10 类 (CIFAR10)。
    """
    def __init__(self, num_classes=10):
        super(ResNetCIFAR10, self).__init__()
        # 加载官方预训练 ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 替换最后一层全连接，以适配 CIFAR10 的 10 类
        in_features = self.model.fc.in_features
        self.linear = nn.Linear(in_features, num_classes)
        self.model.fc = self.linear
    
    def forward(self, x):
        # 返回 [B, 10] logits
        return self.model(x)

    
    def get_backbone(self):
        """
        获取去掉最后一层全连接的部分，用作“特征提取”。
        """
        backbone_layers = list(self.model.children())[:-1]  # 去掉 fc
        backbone = nn.Sequential(*backbone_layers)
        return backbone

@torch.no_grad()
def compute_confidence(model, x):
    """
    给定图像 x (B,3,H,W)，使用分类器 model 前向传播，并计算其对“预测类别”的最大置信度(softmax后最大值)。
    return: shape [B], 每张图像的最大置信度
    """
    model.eval()
    logits = model(x)                      # [B, 10]
    probs = torch.softmax(logits, dim=1)   # [B, 10]
    conf, _ = probs.max(dim=1)             # [B]
    return conf

def train_classifier(model, train_loader, val_loader, device, epochs=2, lr=1e-3):
    """
    简单示例：微调 ResNet18 用于 CIFAR10 分类，并训练若干 epoch。
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # 训练过程添加进度条
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_classifier(model, val_loader, device, criterion)
        print(f"[Classifier] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    return model

@torch.no_grad()
def evaluate_classifier(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    for images, labels in tqdm(data_loader, desc="Evaluating Classifier", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(data_loader.dataset)

# ------------------------------------------------------------------------------------
# 2. 决策网络：从“提取到的128维特征” -> 输出是否需要上传原图(二元决策)
# ------------------------------------------------------------------------------------
class DecisionNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(DecisionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # 输出一个标量，sigmoid后表示概率p(需要原图=1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------------
# 3. 特征提取器：将 ResNet backbone 输出 -> 映射到128维(边缘轻量表示)
# ------------------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone  # [B, 512, 1,1] after avgpool
        # backbone的输出一般是 [B,512,1,1]，flatten之后就是512
        
        
        # 冻结 backbone，以模拟边缘端只做推理
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        x: [B,3,H,W]
        return: [B, out_dim]
        """
        with torch.no_grad():
            feats = self.backbone(x)  # [B,512,1,1]
        feats = feats.view(feats.size(0), -1)   # [B,512]
        out = feats                   # [B,out_dim]
        
        return out

# ------------------------------------------------------------------------------------
# 4. 生成决策网络的训练标签：利用分类器对图像进行推断 -> 如果置信度低于阈值 => label=1
# ------------------------------------------------------------------------------------
def generate_decision_labels(classifier, data_loader, device, threshold=0.8):
    """
    遍历数据集，对每张图片计算分类置信度；若 < threshold，则 label=1(需要原图)，否则0。
    返回: (features_tensor, labels_tensor)，可用来训练决策网络
    """
    # 首先构建一个特征提取器
    backbone = classifier.get_backbone()
    feature_extractor = FeatureExtractor(backbone).to(device)
    
    features_list = []
    labels_list = []
    
    classifier.eval()
    feature_extractor.eval()
    
    for images, _ in tqdm(data_loader, desc="Generating decision labels"):
        images = images.to(device)
        # 1) 计算分类器置信度 (边缘端假设)
        confs = compute_confidence(classifier, images)  # [B]
        # 2) 生成决策网络的标签
        #    label=1  => 需要原图
        #    label=0  => 不需要
        labels = (confs < threshold).float()            # [B]
        
        # 3) 提取128维特征
        feats = feature_extractor(images)               # [B,128]
        
        features_list.append(feats.cpu())
        labels_list.append(labels.cpu())
    
    features_tensor = torch.cat(features_list, dim=0)  # [N,128]
    labels_tensor = torch.cat(labels_list, dim=0)        # [N]
    
    return features_tensor, labels_tensor

# ------------------------------------------------------------------------------------
# 5. 训练决策网络
# ------------------------------------------------------------------------------------
def train_decision_net(decision_net, features, labels, device, epochs=5, lr=1e-3, batch_size=64):
    """
    用离线生成的 (features, labels) 来训练决策网络。
    labels=1 表示需要原图，0 表示不需要。
    """
    decision_net = decision_net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(decision_net.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True )
    
    for epoch in range(1, epochs+1):
        decision_net.train()
        running_loss = 0.0
        
        for x, y in tqdm(loader, desc=f"Training DecisionNet Epoch {epoch}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = decision_net(x).squeeze(1)  # [B]
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # 保留计算图
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_loss = running_loss / len(loader.dataset)
        print(f"[DecisionNet] Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f}")
    return decision_net

# ------------------------------------------------------------------------------------
# 6. 可视化评估：混淆矩阵、ROC、AUC
# ------------------------------------------------------------------------------------
def evaluate_decision_net(decision_net, features, labels, device):
    """
    评估决策网络的性能，并可视化混淆矩阵、ROC曲线等。
    """
    decision_net.eval()
    features = features.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        logits = decision_net(features).squeeze(1)  # [N]
        probs = torch.sigmoid(logits)
    
    probs_cpu = probs.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    
    # 二值化预测
    pred_binary = (probs_cpu > 0.5).astype(int)
    
    # 混淆矩阵
    cm = confusion_matrix(labels_cpu, pred_binary)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred=0", "Pred=1"],
                yticklabels=["True=0", "True=1"])
    plt.title("Confusion Matrix for DecisionNet")
    plt.savefig("confusion_matrix.png")
    plt.show()
    
    # ROC & AUC
    fpr, tpr, _ = roc_curve(labels_cpu, probs_cpu)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='red', label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1], color='blue', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - DecisionNet")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()

# ------------------------------------------------------------------------------------
# 7. 边云协同流程模拟：无人机 & 云服务器
# ------------------------------------------------------------------------------------
class Drone:
    """
    无人机：
     - 使用与训练时一致的特征提取方式(固定backbone+fc->128维)，
     - 同时可获取分类置信度(若需要)。这里只演示特征上传给云端。
    """
    def __init__(self, backbone, device):
        self.device = device
        # 用来做128维特征提取
        self.feature_extractor = FeatureExtractor(backbone).to(device)
        self.current_image = None
    
    def capture_image(self, image):
        self.current_image = image  # [3,H,W]
    
    @torch.no_grad()
    def get_feature(self):
        """
        获取当前图像的128维特征
        """
        if self.current_image is None:
            return None
        self.feature_extractor.eval()
        img_batch = self.current_image.unsqueeze(0).to(self.device)  # [1,3,H,W]
        feature = self.feature_extractor(img_batch)  # [1,128]
        return feature.squeeze(0)  # [128]
    
    def send_feature_to_cloud(self, cloud_server):
        feature = self.get_feature()
        if feature is not None:
            cloud_server.receive_feature(feature, self)
    
    def send_raw_image_to_cloud(self, cloud_server):
        """
        当云端请求原图时，发送
        """
        print("Drone: >>> 上传原图到云端!")
        cloud_server.receive_raw_image(self.current_image)

class CloudServer:
    """
    云服务器：接收 128维特征 -> 通过决策网络判断 -> 请求或不请求原图
    """
    def __init__(self, decision_net, device):
        self.decision_net = decision_net.to(device)
        self.device = device
    
    def receive_feature(self, feature, drone):
        self.decision_net.eval()
        with torch.no_grad():
            feature = feature.unsqueeze(0).to(self.device)  # [1,128]
            score = torch.sigmoid(self.decision_net(feature))  # [1,1]
            score_val = score.item()
        
        print(f"Cloud: decision score={score_val:.4f}", end=" | ")
        if score_val > 0.5:
            print("需要原图 -> 向无人机请求")
            drone.send_raw_image_to_cloud(self)
        else:
            print("不需要原图(置信度足够)")
    
    def receive_raw_image(self, image):
        print(f"Cloud: 已收到原图, 尺寸={tuple(image.shape)}, 在云端执行高阶任务...\n")

# ------------------------------------------------------------------------------------
# 8. 模拟在线推理过程
# ------------------------------------------------------------------------------------
def simulate_edge_cloud_inference(drone, cloud_server, test_loader, device, num_images=5):
    print("\n=== 开始『边云协同』在线仿真 ===")
    count = 0
    # 遍历测试集，并使用进度条
    for images, _ in tqdm(test_loader, desc="Simulating edge-cloud inference"):
        for img in images:
            if count >= num_images:
                return
            # 无人机采集图像
            drone.capture_image(img)
            # 无人机端 -> 提取128维特征 -> 上传云端
            print(f"[Image #{count+1}] 无人机先上传特征 -> 云端决策")
            drone.send_feature_to_cloud(cloud_server)
            count += 1

# ------------------------------------------------------------------------------------
# 9. 额外模块：比较『有决策网络』vs『无决策网络』效果
# ------------------------------------------------------------------------------------
def compare_scenarios(classifier, decision_net, test_loader, device, threshold=0.8):
    classifier.eval()
    decision_net.eval()
    
    total_samples = 0
    correct_A, correct_B, correct_C = 0, 0, 0
    raw_count_A, raw_count_B, raw_count_C = 0, 0, 0
    
    backbone = classifier.get_backbone()
    feature_extractor = FeatureExtractor(backbone, out_dim=128).to(device)
    
    for images, labels in tqdm(test_loader, desc="Comparing scenarios"):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        total_samples += batch_size
        
        # 场景A: 始终上传原图 => 云端推理
        outputs_A = classifier(images)
        preds_A = outputs_A.argmax(dim=1)
        correct_A += (preds_A == labels).sum().item()
        raw_count_A += batch_size
        
        # 场景B: 从不上传原图 => 边缘端推理
        outputs_B = classifier(images)
        preds_B = outputs_B.argmax(dim=1)
        correct_B += (preds_B == labels).sum().item()
        
        # 场景C: 使用决策网络决定是否上传原图
        features = feature_extractor(images)
        decision_scores = torch.sigmoid(decision_net(features)).squeeze(1)
        need_raw = decision_scores > 0.5
        
        preds_C = []
        for i in range(batch_size):
            if need_raw[i]:
                raw_count_C += 1
                single_img = images[i].unsqueeze(0)
                out_cloud = classifier(single_img)
                pred_cloud = out_cloud.argmax(dim=1)
                preds_C.append(pred_cloud.item())
            else:
                single_img = images[i].unsqueeze(0)
                out_edge = classifier(single_img)
                pred_edge = out_edge.argmax(dim=1)
                preds_C.append(pred_edge.item())
        preds_C = torch.tensor(preds_C).to(device)
        correct_C += (preds_C == labels).sum().item()
    
    accuracy_A = correct_A / total_samples
    accuracy_B = correct_B / total_samples
    accuracy_C = correct_C / total_samples
    
    raw_upload_ratio_A = raw_count_A / total_samples
    raw_upload_ratio_B = 0.0
    raw_upload_ratio_C = raw_count_C / total_samples
    
    results = {
        'A': (accuracy_A, raw_upload_ratio_A),
        'B': (accuracy_B, raw_upload_ratio_B),
        'C': (accuracy_C, raw_upload_ratio_C),
    }
    return results


def plot_comparison(results):
    """
    可视化三个场景(A/B/C)的accuracy和raw_upload_ratio对比柱状图
    results: dict, {'A':(accA, ratioA), 'B':(accB, ratioB), 'C':(accC, ratioC)}
    """
    scenarios = ['A-AlwaysRaw', 'B-NeverRaw', 'C-DecisionNet']
    accuracies = [results['A'][0], results['B'][0], results['C'][0]]
    raw_ratios = [results['A'][1], results['B'][1], results['C'][1]]
    
    # --------- 画Accuracy对比 ---------
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.bar(scenarios, accuracies, color=['blue','green','red'])
    plt.ylabel("Accuracy")
    plt.ylim([0,1])
    plt.title("Accuracy Comparison")
    
    # --------- 画Raw Upload Ratio对比 ---------
    plt.subplot(1,2,2)
    plt.bar(scenarios, raw_ratios, color=['blue','green','red'])
    plt.ylabel("Raw Upload Ratio")
    plt.ylim([0,1])
    plt.title("Raw Upload Ratio Comparison")
    
    plt.tight_layout()
    plt.savefig("scenario_comparison.png")
    plt.show()

# ------------------------------------------------------------------------------------
# 10. 主函数
# ------------------------------------------------------------------------------------
def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # -- 数据集准备：CIFAR10 (32x32) --
    #    ResNet18预训练模型通常适配224x224，这里简单用Resize到224
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    
    # 请确保cifar数据已放在对应目录下(可改成 download=True)
    cifar_trainval = datasets.CIFAR10(root=os.path.join(current_path, 'cifar-10-python/'), 
                                      train=True, download=False, transform=transform)
    cifar_test = datasets.CIFAR10(root=os.path.join(current_path, 'cifar-10-python/'), 
                                  train=False, download=False, transform=transform)
    
    # 划分训练/验证集
    train_size = int(0.8 * len(cifar_trainval))
    val_size = len(cifar_trainval) - train_size
    cifar_train, cifar_val = random_split(cifar_trainval, [train_size, val_size])
    
    train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(cifar_val,   batch_size=64, shuffle=False, num_workers=2)
    test_loader  = DataLoader(cifar_test,  batch_size=32, shuffle=False, num_workers=2)
    
    # ========== 步骤1：训练“分类器”获取分类置信度 ========== #
    print("==== 训练分类器(微调ResNet18) ====")
    classifier = ResNetCIFAR10(num_classes=10)
    # 为演示起见，只训练1个 epoch，实际可更多
    classifier = train_classifier(classifier, train_loader, val_loader, device, epochs=1, lr=1e-3)
    
    # ========== 步骤2：为训练决策网络生成标签(label=1表示置信度不足，需要原图) ========== #
    print("\n==== 基于分类置信度生成决策网络的训练标签 ====")
    # 在训练集和验证集一起生成特征和标签，便于训练决策网络
    
    combined_dataset = torch.utils.data.ConcatDataset([cifar_train, cifar_val])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    features_tensor, labels_tensor = generate_decision_labels(classifier, combined_loader, device, threshold=0.8)
    print(f"特征张量大小: {features_tensor.shape}, 标签张量大小: {labels_tensor.shape}")
    positive_ratio = (labels_tensor==1).float().mean().item()
    print(f"有 {positive_ratio*100:.2f}% 样本需要原图(置信度低于0.8)")
    
    # ========== 步骤3：训练决策网络(输入128维 -> 输出是否需原图) ========== #
    print("\n==== 训练决策网络 DecisionNet ====")
    decision_net = DecisionNet(input_dim=512, hidden_dim=64)
    decision_net = train_decision_net(decision_net, features_tensor, labels_tensor, device, 
                                      epochs=1, lr=1e-3, batch_size=64)
    
    # ========== 步骤4：可视化评估决策网络 ========== #
    print("\n==== 评估决策网络 ====")
    evaluate_decision_net(decision_net, features_tensor, labels_tensor, device)
    
    # ========== 步骤5：模拟线上场景：无人机只提取128维特征 -> 云端根据决策网络判断 ========== #
    print("\n==== 模拟边云协同场景 ====")
    # 构建无人机：使用分类器的backbone (并冻结)
    backbone = classifier.get_backbone()
    drone = Drone(backbone, device)
    
    # 构建云服务器
    cloud_server = CloudServer(decision_net, device)
    
    simulate_edge_cloud_inference(drone, cloud_server, test_loader, device, num_images=10)
    
    # ========== 步骤6：对比『有决策网络』 vs 『无决策网络』 ========== #
    print("\n==== 对比场景A/B/C 的 效果 ====")
    # 使用测试集对比三种场景
    test_loader_compare = DataLoader(cifar_test, batch_size=64, shuffle=False, num_workers=2)
    results = compare_scenarios(classifier, decision_net, test_loader_compare, device, threshold=0.8)
    # results 是字典 {'A':(accA, ratioA), 'B':(accB, ratioB), 'C':(accC, ratioC)}
    
    # 打印比较结果
    print("场景A (Always Raw):    accuracy = {:.3f}, raw ratio = {:.3f}".format(*results['A']))
    print("场景B (Never Raw):     accuracy = {:.3f}, raw ratio = {:.3f}".format(*results['B']))
    print("场景C (Decision Net):  accuracy = {:.3f}, raw ratio = {:.3f}".format(*results['C']))
    
    # 可视化对比
    plot_comparison(results)

if __name__ == "__main__":
    main()