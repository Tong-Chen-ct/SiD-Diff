# -*- coding: utf-8 -*-
#model-8后续调试代码
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 自定义 GELU 激活函数
def gelu(x):
    return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))

# ==================== 位置编码层 ====================
class SimplifiedRegionalEncoding(layers.Layer):
    def __init__(self, embed_dim=8, bin_size=5000):
        super(SimplifiedRegionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.bin_size = bin_size
        self.distance_embed = layers.Dense(
            embed_dim, activation='tanh',
            kernel_regularizer=regularizers.l2(0.01),
            name='distance_embed'
        )
   
    def call(self, x):
        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
       
        i_indices = tf.cast(tf.range(h), tf.float32)
        j_indices = tf.cast(tf.range(w), tf.float32)
        i_grid = tf.tile(tf.reshape(i_indices, [-1, 1]), [1, w])
        j_grid = tf.tile(tf.reshape(j_indices, [1, -1]), [h, 1])
       
        genomic_distance = tf.abs(i_grid - j_grid) * self.bin_size
        log_distance = tf.math.log1p(genomic_distance)
        max_log_distance = tf.math.log1p(tf.cast(h * self.bin_size, tf.float32))
        normalized_distance = log_distance / (max_log_distance + 1e-8)
       
        distance_map = tf.expand_dims(normalized_distance, axis=-1)
        distance_map = tf.tile(tf.expand_dims(distance_map, 0), [batch_size, 1, 1, 1])
       
        return self.distance_embed(distance_map)
        
class SymmetryAwareEncoding(layers.Layer):
    """
    对称性编码 - 捕获Hi-C矩阵的上下三角区别
    理论上应对称，实际可能有技术偏差
    """
    def __init__(self, embed_dim):
        super(SymmetryAwareEncoding, self).__init__()
        self.upper_embedding = self.add_weight(
            shape=(1, 1, 1, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='upper_triangle_embed'
        )
        self.lower_embedding = self.add_weight(
            shape=(1, 1, 1, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='lower_triangle_embed'
        )
        self.diagonal_embedding = self.add_weight(
            shape=(1, 1, 1, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='diagonal_embed'
        )
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        
        # 创建mask
        i_indices = tf.range(h)
        j_indices = tf.range(w)
        i_grid = tf.tile(tf.reshape(i_indices, [-1, 1]), [1, w])
        j_grid = tf.tile(tf.reshape(j_indices, [1, -1]), [h, 1])
        
        upper_mask = tf.cast(j_grid > i_grid, tf.float32)
        lower_mask = tf.cast(j_grid < i_grid, tf.float32)
        diag_mask = tf.cast(j_grid == i_grid, tf.float32)
        
        # 构建位置编码
        upper_mask = tf.reshape(upper_mask, [1, h, w, 1])
        lower_mask = tf.reshape(lower_mask, [1, h, w, 1])
        diag_mask = tf.reshape(diag_mask, [1, h, w, 1])
        
        pos_encoding = (upper_mask * self.upper_embedding + 
                       lower_mask * self.lower_embedding + 
                       diag_mask * self.diagonal_embedding)
        
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1, 1])
        
        return pos_encoding
        
class DiagonalDistanceEncoding(layers.Layer):
    """
    对角线距离编码 - Hi-C矩阵最核心的位置特征
    距离对角线越远 = 基因组距离越大 = 长程互作
    """
    def __init__(self, embed_dim):
        super(DiagonalDistanceEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.distance_embedding = layers.Dense(embed_dim, name='diagonal_embed')
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        
        # 创建对角线距离矩阵
        i_indices = tf.cast(tf.range(h), tf.float32)
        j_indices = tf.cast(tf.range(w), tf.float32)
        
        i_grid = tf.tile(tf.reshape(i_indices, [-1, 1]), [1, w])
        j_grid = tf.tile(tf.reshape(j_indices, [1, -1]), [h, 1])
        
        # 计算对角线距离
        diagonal_distance = tf.abs(i_grid - j_grid)
        
        # Log scale归一化（Hi-C信号呈幂律衰减）
        log_distance = tf.math.log1p(diagonal_distance)
        max_log_distance = tf.math.log1p(tf.cast(h, tf.float32))
        normalized_distance = log_distance / (max_log_distance + 1e-8)
        
        # 扩展维度
        distance_map = tf.expand_dims(normalized_distance, axis=-1)  # (h, w, 1)
        distance_map = tf.tile(
            tf.expand_dims(distance_map, 0), 
            [batch_size, 1, 1, 1]
        )  # (batch, h, w, 1)
        
        # 嵌入到高维空间
        distance_embed = self.distance_embedding(distance_map)
        
        return distance_embed


# ==================== Transformer层 ====================
class LightweightTransformer(layers.Layer):
    def __init__(self, embed_dim=16, num_heads=2, ff_dim=32, dropout_rate=0.3):
        super(LightweightTransformer, self).__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation=gelu, kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.01))
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
   
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
   
# ==================== 主模型 ====================
class SimplifiedHiCNet(Model):
    def __init__(self, input_shape=(100, 100, 1), embed_dim=16, num_heads=2,
                 bin_size=5000, dropout_rate=0.3):
        super(SimplifiedHiCNet, self).__init__()
        
        # 保持原有的特征提取层不变
        self.regional_encoding = SimplifiedRegionalEncoding(embed_dim=8, bin_size=bin_size)
        
        self.conv1 = layers.Conv2D(16, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(gelu)
        self.pool1 = layers.MaxPooling2D(2, strides=2)
        self.dropout_conv1 = layers.Dropout(dropout_rate)
        
        self.conv2 = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(gelu)
        self.pool2 = layers.MaxPooling2D(2, strides=2)
        self.dropout_conv2 = layers.Dropout(dropout_rate)
        
        self.reshape_to_seq = layers.Reshape((-1, 32))
        self.proj = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.01))
        
        self.transformer1 = LightweightTransformer(
            embed_dim=embed_dim, num_heads=num_heads,
            ff_dim=embed_dim*2, dropout_rate=dropout_rate
        )
        
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # ---【修改部分开始】---
        # 融合层
        self.fusion_dense1 = layers.Dense(64, activation=gelu, kernel_regularizer=regularizers.l2(0.01))
        self.dropout_fusion = layers.Dropout(dropout_rate)
        
        # 分类头 (Classifier Head)
        # 直接输出 1 个值，经过 Sigmoid 变为概率 (0~1)
        # 0 代表相似 (Label 0)，1 代表不相似 (Label 1)
        self.classifier = layers.Dense(1, activation='sigmoid', name='probability_output')
        # ---【修改部分结束】---
    
    def process_single_matrix(self, x, training=False):
        pos_encoding = self.regional_encoding(x)
        x = tf.concat([x, pos_encoding], axis=-1)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout_conv1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout_conv2(x, training=training)
        
        x = self.reshape_to_seq(x)
        x = self.proj(x)
        x = self.transformer1(x, training=training)
        return x
    
    def call(self, inputs, training=False):
        x1, x2 = inputs
        
        feat1 = self.process_single_matrix(x1, training)
        feat2 = self.process_single_matrix(x2, training)
        
        #有交叉注意力
        cross_feat1 = self.cross_attention(query=feat1, key=feat2, value=feat2, training=training)
        cross_feat2 = self.cross_attention(query=feat2, key=feat1, value=feat1, training=training)
        
        pooled1 = self.global_pool(cross_feat1)
        pooled2 = self.global_pool(cross_feat2)
        abs_diff = self.global_pool(tf.abs(feat1 - feat2))
        '''
        # 无交叉注意力，直接使用原始feat1和feat2
        pooled1 = self.global_pool(feat1)  # 替换cross_feat1
        pooled2 = self.global_pool(feat2)  # 替换cross_feat2
        abs_diff = self.global_pool(tf.abs(feat1 - feat2))
        '''
        # 融合特征
        diff = tf.concat([pooled1, pooled2, abs_diff], axis=-1)
        diff = self.fusion_dense1(diff)
        diff = self.dropout_fusion(diff, training=training)
        
        # 输出概率
        output = self.classifier(diff)
        
        return output

# ==================== 修复后的损失函数 ====================
class SimplifiedContrastiveLoss(tf.keras.losses.Loss):
    """修复后的对比损失 - 计算两个embedding之间的实际距离"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
   
    def call(self, y_true, y_pred):
        # y_pred 是模型输出的embedding，shape: (batch, embed_dim)
        # 这里我们将其视为特征差异，计算其L2范数作为距离
        euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1) + 1e-8)
        
        # 裁剪距离避免数值溢出
        euclidean_distance = tf.clip_by_value(euclidean_distance, 0, 100)
        
        # 相似对 (y_true=0): 最小化距离
        loss_similar = (1 - y_true) * tf.square(euclidean_distance)
        
        # 不同对 (y_true=1): 距离应大于margin
        loss_diff = y_true * tf.square(tf.maximum(self.margin - euclidean_distance, 0))
        
        return tf.reduce_mean(loss_similar + loss_diff)
        
class AdaptiveDistillationLoss(tf.keras.losses.Loss):
    """
    修正后的二分类蒸馏损失
    Hard Loss: Binary Cross Entropy (分类准确性)
    Soft Loss: Mean Squared Error (让学生的概率输出逼近教师的概率输出)
    """
    
    def __init__(self, initial_alpha=0.7, final_alpha=0.3, temperature=1.0):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.current_alpha = initial_alpha
        # 在 MSE 模式下 temperature 不是必须的，但保留接口以兼容代码
        self.temperature = temperature
        
        # 硬标签损失：二分类交叉熵
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        # 软标签损失：均方误差
        self.mse_loss = tf.keras.losses.MeanSquaredError()
    
    def update_alpha(self, progress):
        """根据训练进度更新alpha"""
        self.current_alpha = self.initial_alpha - (self.initial_alpha - self.final_alpha) * progress
        
    def call(self, y_true, y_pred_student, y_pred_teacher=None):
        # 1. Hard Loss: 真实标签 vs 学生预测
        # y_true: [batch, 1], y_pred_student: [batch, 1] (概率值)
        hard_loss = self.bce_loss(y_true, y_pred_student)
        
        # 如果没有教师模型（Stage 1），只返回 Hard Loss
        if y_pred_teacher is None:
            return hard_loss
        
        # 2. Soft Loss: 教师预测 vs 学生预测
        # 使用 MSE 让学生直接学习教师的置信度
        # stop_gradient 防止梯度传回教师
        soft_loss = self.mse_loss(tf.stop_gradient(y_pred_teacher), y_pred_student)
        
        # 3. 组合损失
        # BCE 和 MSE 的数值范围都在 0~1 之间，直接加权即可，非常稳定
        total_loss = (1.0 - self.current_alpha) * hard_loss + self.current_alpha * soft_loss
        
        return total_loss

# ==================== 数据集和工具函数 ====================
class IndexedNPZHiCDataset(tf.data.Dataset):
    def _generator(npz_file, indices, add_noise=False, noise_factor=0.02):
        data = np.load(npz_file, allow_pickle=True)
        X1 = data['X1'].astype(np.float32)
        X2 = data['X2'].astype(np.float32)
        y = data['y'].astype(np.float32)
       
        for idx in indices:
            img1 = np.expand_dims(X1[idx], axis=-1)
            img2 = np.expand_dims(X2[idx], axis=-1)
            label = y[idx]
           
            if add_noise:
                img1 += np.random.normal(0, noise_factor, img1.shape).astype(np.float32)
                img2 += np.random.normal(0, noise_factor, img2.shape).astype(np.float32)
           
            yield img1, img2, label
   
    def __new__(cls, npz_file, indices, batch_size, add_noise=False, shuffle=True):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            args=[npz_file, indices, add_noise],
            output_signature=(
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(indices), 1000))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_cv_splits(splits_path):
    """加载预定义的交叉验证划分"""
    data = np.load(splits_path, allow_pickle=True)
   
    n_folds = int(data['n_folds'])
    splits = {}
   
    for fold in range(n_folds):
        splits[fold] = {
            'train_idx': data[f'fold_{fold}_train_idx'],
            'val_idx': data[f'fold_{fold}_val_idx'],
            'test_idx': data[f'fold_{fold}_test_idx'],
        }
   
    print(f"已加载 {n_folds} 折交叉验证划分")
    for fold in range(n_folds):
        print(f" Fold {fold+1}: 训练 {len(splits[fold]['train_idx'])}, "
              f"验证 {len(splits[fold]['val_idx'])}, "
              f"测试 {len(splits[fold]['test_idx'])}")
   
    return splits, n_folds
def compute_metrics(model, dataset, device_name):
    """
    计算评估指标 - 概率版
    不需要计算距离，直接使用模型输出的概率值作为 Score
    """
    all_scores = []
    true_labels = []
    
    with tf.device(device_name):
        for input1, input2, labels in dataset:
            # output shape: [batch_size, 1] (值在 0~1 之间)
            output = model([input1, input2], training=False)
            
            # 将 (batch, 1) 展平为 (batch,)
            batch_scores = tf.reshape(output, [-1]).numpy()
            
            all_scores.append(batch_scores)
            true_labels.append(labels.numpy())
    
    # 拼接所有 batch 的结果
    all_scores = np.concatenate(all_scores, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    
    # --- 指标计算 ---
    # Score 越大 (接近1) -> 预测为类别 1 (不相似/差异)
    # Score 越小 (接近0) -> 预测为类别 0 (相似)
    # 这完全符合 sklearn roc_curve 的输入要求
    
    # 1. ROC-AUC
    fpr, tpr, thresholds = roc_curve(true_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # 2. PR-AUC
    pr_auc = average_precision_score(true_labels, all_scores)
    
    # 3. 计算最佳阈值 (Youden's J statistic)
    # = np.argmax(tpr - fpr)
    #threshold = thresholds[optimal_idx]
    threshold = 0.5

    # 4. 基于最佳阈值生成预测标签
    predictions = (all_scores >= threshold).astype(int)
    
    # 5. 计算常规分类指标
    accuracy = np.mean(predictions == true_labels)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold,
        'scores': all_scores,
        'true_labels': true_labels
    }
# ==================== 改进的训练器 ====================
class TwoStageTrainer:
    """添加梯度裁剪和异常检测的训练器"""

    def __init__(self, teacher_model, student_model,
                 raw_npz, curated_npz,
                 train_idx_raw, val_idx_raw, test_idx_raw,
                 train_idx_cur, val_idx_cur, test_idx_cur,
                 batch_size=32, device_name="/GPU:0", outpath="./"):

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.raw_npz = raw_npz
        self.curated_npz = curated_npz

        self.train_idx_raw = train_idx_raw
        self.val_idx_raw = val_idx_raw
        self.test_idx_raw = test_idx_raw
        self.train_idx_cur = train_idx_cur
        self.val_idx_cur = val_idx_cur
        self.test_idx_cur = test_idx_cur

        self.batch_size = batch_size
        self.device_name = device_name
        self.outpath = outpath
        os.makedirs(outpath, exist_ok=True)

        self.history = {
            'stage1': {'train_loss': [], 'val_loss': [], 'val_metrics': []},
            'stage2': {'train_loss': [], 'val_loss': [], 'val_metrics': []},
        }

    def stage1_pretrain(self, epochs=100, learning_rate=0.0003):
        """阶段1: 添加梯度裁剪"""
        print("\n" + "=" * 60)
        print("阶段 1: 学生模型预训练（原始数据）")
        print("=" * 60)

        train_dataset = IndexedNPZHiCDataset(
            self.raw_npz, self.train_idx_raw, self.batch_size, add_noise=True
        )
        val_dataset = IndexedNPZHiCDataset(
            self.raw_npz, self.val_idx_raw, self.batch_size, add_noise=False
        )

        optimizer = Adam(learning_rate, clipnorm=1.0)  # 添加梯度裁剪
        #loss_fn = SimplifiedContrastiveLoss(margin=1.0)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nStage 1 - Epoch {epoch + 1}/{epochs}")

            train_loss = 0.0
            train_steps = 0
            for x1, x2, y in train_dataset:
                with tf.GradientTape() as tape:
                    preds = self.student_model([x1, x2], training=True)
                    loss = loss_fn(y, preds)
                    
                    # 检查异常损失
                    if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                        print(f"警告: 检测到异常损失值 {loss.numpy()}，跳过此batch")
                        continue

                grads = tape.gradient(loss, self.student_model.trainable_variables)
                
                # 检查梯度
                if any(tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)) 
                       for g in grads if g is not None):
                    print("警告: 检测到异常梯度，跳过此batch")
                    continue
                
                optimizer.apply_gradients(zip(grads, self.student_model.trainable_variables))

                train_loss += float(loss.numpy())
                train_steps += 1

            if train_steps == 0:
                print("错误: 所有batch都失败了！")
                break

            avg_train_loss = train_loss / train_steps

            # 验证
            val_loss_sum = 0.0
            val_steps = 0
            for x1, x2, y in val_dataset:
                preds = self.student_model([x1, x2], training=False)
                loss = loss_fn(y, preds)
                
                if not (tf.math.is_nan(loss) or tf.math.is_inf(loss)):
                    val_loss_sum += float(loss.numpy())
                    val_steps += 1
            
            if val_steps == 0:
                print("警告: 验证集所有batch损失异常")
                continue
                
            avg_val_loss = val_loss_sum / val_steps

            val_metrics = compute_metrics(self.student_model, val_dataset, self.device_name)

            print(f" 训练损失: {avg_train_loss:.4f}")
            print(f" 验证损失: {avg_val_loss:.4f}")

            self.history['stage1']['train_loss'].append(avg_train_loss)
            self.history['stage1']['val_loss'].append(avg_val_loss)
            self.history['stage1']['val_metrics'].append(val_metrics)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.student_model.save_weights(os.path.join(self.outpath, "student_stage1_best.ckpt"))
                patience_counter = 0
                print(f" ? val_loss改进 → 保存模型，最佳: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f" ? 无改进 (Patience: {patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(" 早停触发！")
                    break

        self.student_model.load_weights(os.path.join(self.outpath, "student_stage1_best.ckpt"))
        print(f"\n? 阶段1完成! 最佳val_loss: {best_val_loss:.4f}")
        return best_val_loss
    '''
    def stage2_distillation(self, epochs=100, learning_rate=0.0001,  # 降低学习率
                            initial_alpha=0.7, final_alpha=0.3, temperature=1.5):
        """阶段2: 改进的蒸馏训练"""
        print("\n" + "=" * 60)
        print("阶段 2: 知识蒸馏（教师→学生，最终模型）")
        print("=" * 60)
        print(f"蒸馏参数: alpha={initial_alpha}→{final_alpha}, temp={temperature}")
        print(f"学习率: {learning_rate} (已降低以提高稳定性)")

        # 训练教师
        print("\n训练教师模型...")
        self._train_teacher(epochs=100, learning_rate=0.0003)

        train_dataset_raw = IndexedNPZHiCDataset(
            self.raw_npz, self.train_idx_raw, self.batch_size, add_noise=False
        )
        val_dataset_raw = IndexedNPZHiCDataset(
            self.raw_npz, self.val_idx_raw, self.batch_size, add_noise=False
        )

        optimizer = Adam(learning_rate, clipnorm=1.0)  # 梯度裁剪
        distill_loss_fn = AdaptiveDistillationLoss(
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            temperature=temperature
        )

        best_val_roc = 0.0
        patience = 20
        patience_counter = 0
        consecutive_failures = 0  # 连续失败计数

        for epoch in range(epochs):
            print(f"\nStage 2 - Epoch {epoch + 1}/{epochs}")

            progress = epoch / float(max(epochs, 1))
            distill_loss_fn.update_alpha(progress)
            print(f" 当前alpha: {distill_loss_fn.current_alpha:.3f}")

            train_loss = 0.0
            train_steps = 0
            
            for x1_raw, x2_raw, y_raw in train_dataset_raw:
                with tf.GradientTape() as tape:
                    pred_student = self.student_model([x1_raw, x2_raw], training=True)
                    pred_teacher = self.teacher_model([x1_raw, x2_raw], training=False)
                    loss = distill_loss_fn(y_raw, pred_student, pred_teacher)
                    
                    # 异常检测
                    if tf.math.is_nan(loss) or tf.math.is_inf(loss) or loss > 10000:
                        print(f"警告: 异常损失 {loss.numpy()}，跳过")
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            print("错误: 连续失败过多，终止训练")
                            return best_val_roc
                        continue
                    
                    consecutive_failures = 0  # 重置失败计数

                grads = tape.gradient(loss, self.student_model.trainable_variables)
                
                # 检查梯度
                if any(tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)) 
                       for g in grads if g is not None):
                    print("警告: 异常梯度，跳过")
                    continue
                
                optimizer.apply_gradients(zip(grads, self.student_model.trainable_variables))

                train_loss += float(loss.numpy())
                train_steps += 1

            if train_steps == 0:
                print("错误: epoch中所有batch都失败")
                break

            avg_train_loss = train_loss / train_steps

            # 验证
            val_loss_sum = 0.0
            val_steps = 0
            for x1_val, x2_val, y_val in val_dataset_raw:
                pred_student_val = self.student_model([x1_val, x2_val], training=False)
                val_loss = distill_loss_fn(y_val, pred_student_val)
                
                if not (tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss) or val_loss > 10000):
                    val_loss_sum += float(val_loss.numpy())
                    val_steps += 1
            
            if val_steps == 0:
                print("警告: 验证集所有batch异常")
                continue
                
            avg_val_loss = val_loss_sum / val_steps

            val_metrics = compute_metrics(self.student_model, val_dataset_raw, self.device_name)

            print(f" 训练损失: {avg_train_loss:.4f}")
            print(f" 验证损失: {avg_val_loss:.4f}")
            print(f" 验证ROC-AUC: {val_metrics['roc_auc']:.4f}")

            self.history['stage2']['train_loss'].append(avg_train_loss)
            self.history['stage2']['val_loss'].append(avg_val_loss)
            self.history['stage2']['val_metrics'].append(val_metrics)

            if val_metrics['roc_auc'] > best_val_roc:
                best_val_roc = val_metrics['roc_auc']
                self.student_model.save_weights(os.path.join(self.outpath, "student_stage2_best.ckpt"))
                patience_counter = 0
                print(f" ? ROC-AUC改进 → 保存模型，最佳: {best_val_roc:.4f}")
            else:
                patience_counter += 1
                print(f" ? 无改进 (Patience: {patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(" 早停触发！")
                    break

        self.student_model.load_weights(os.path.join(self.outpath, "student_stage2_best.ckpt"))
        print(f"\n? 阶段2完成! 最佳ROC-AUC: {best_val_roc:.4f}")
        return best_val_roc
        '''

    def stage2_distillation(self, epochs=100, learning_rate=0.0001, initial_alpha=0.7, final_alpha=0.3,
                            temperature=1.5):
        """阶段2: 改进的蒸馏训练"""
        print("\n" + "=" * 60)
        print("阶段2: 知识蒸馏（教师→学生，最终模型）")
        print("=" * 60)
        print(f"蒸馏参数: alpha={initial_alpha}→{final_alpha}, temp={temperature}")
        print(f"学习率: {learning_rate} (已降低以提高稳定性)")

        # 训练教师
        print("\n训练教师模型...")
        self._train_teacher(epochs=100, learning_rate=0.0003)

        train_dataset_raw = IndexedNPZHiCDataset(
            self.raw_npz, self.train_idx_raw, self.batch_size, add_noise=False
        )
        val_dataset_raw = IndexedNPZHiCDataset(
            self.raw_npz, self.val_idx_raw, self.batch_size, add_noise=False
        )

        optimizer = Adam(learning_rate, clipnorm=1.0)
        distill_loss_fn = AdaptiveDistillationLoss(
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            temperature=temperature
        )

        best_val_roc = 0.0
        patience = 30
        patience_counter = 0
        consecutive_failures = 0

        for epoch in range(epochs):
            print(f"\nStage 2 - Epoch {epoch + 1}/{epochs}")

            progress = epoch / float(max(epochs, 1))
            distill_loss_fn.update_alpha(progress)
            print(f"当前alpha: {distill_loss_fn.current_alpha:.3f}")

            # ========== 训练部分 ==========
            train_loss = 0.0
            train_steps = 0

            for x1_raw, x2_raw, y_raw in train_dataset_raw:
                with tf.GradientTape() as tape:
                    pred_student = self.student_model([x1_raw, x2_raw], training=True)
                    pred_teacher = self.teacher_model([x1_raw, x2_raw], training=False)
                    loss = distill_loss_fn(y_raw, pred_student, pred_teacher)

                    if tf.math.is_nan(loss) or tf.math.is_inf(loss) or loss > 10000:
                        print(f"警告: 异常损失 {loss.numpy()}，跳过")
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            print("错误: 连续失败过多，终止训练")
                            return best_val_roc
                        continue

                    consecutive_failures = 0

                grads = tape.gradient(loss, self.student_model.trainable_variables)

                if any(tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))
                       for g in grads if g is not None):
                    print("警告: 异常梯度，跳过")
                    continue

                optimizer.apply_gradients(zip(grads, self.student_model.trainable_variables))

                train_loss += float(loss.numpy())
                train_steps += 1

            if train_steps == 0:
                print("错误: epoch中所有batch都失败")
                break

            avg_train_loss = train_loss / train_steps

            # ========== 验证部分（修改后）==========
            val_loss_sum = 0.0
            val_steps = 0
            for x1_val, x2_val, y_val in val_dataset_raw:
                pred_student_val = self.student_model([x1_val, x2_val], training=False)
                # ✅ 添加教师模型预测
                pred_teacher_val = self.teacher_model([x1_val, x2_val], training=False)
                # ✅ 使用完整的蒸馏损失（与训练时一致）
                val_loss = distill_loss_fn(y_val, pred_student_val, pred_teacher_val)

                if not (tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss) or val_loss > 10000):
                    val_loss_sum += float(val_loss.numpy())
                    val_steps += 1

            if val_steps == 0:
                print("警告: 验证集所有batch异常")
                continue
            avg_val_loss = val_loss_sum / val_steps

            val_metrics = compute_metrics(self.student_model, val_dataset_raw, self.device_name)

            print(f" 训练损失: {avg_train_loss:.4f}")
            print(f" 验证损失: {avg_val_loss:.4f}")
            print(f" 验证ROC-AUC: {val_metrics['roc_auc']:.4f}")

            self.history['stage2']['train_loss'].append(avg_train_loss)
            self.history['stage2']['val_loss'].append(avg_val_loss)
            self.history['stage2']['val_metrics'].append(val_metrics)

            if val_metrics['roc_auc'] > best_val_roc:
                best_val_roc = val_metrics['roc_auc']
                self.student_model.save_weights(os.path.join(self.outpath, "student_stage2_best.ckpt"))
                patience_counter = 0
                print(f" ✓ ROC-AUC改进→保存模型，最佳: {best_val_roc:.4f}")
            else:
                patience_counter += 1
                print(f" ✗ 无改进 (Patience: {patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(" 早停触发！")
                    break

        self.student_model.load_weights(os.path.join(self.outpath, "student_stage2_best.ckpt"))
        print(f"\n✓阶段2完成! 最佳ROC-AUC: {best_val_roc:.4f}")
        return best_val_roc

    def _train_teacher(self, epochs=100, learning_rate=0.0003):
        """训练教师模型"""
        train_dataset = IndexedNPZHiCDataset(
            self.curated_npz, self.train_idx_cur, self.batch_size, add_noise=True
        )
        val_dataset = IndexedNPZHiCDataset(
            self.curated_npz, self.val_idx_cur, self.batch_size, add_noise=False
        )

        optimizer = Adam(learning_rate, clipnorm=1.0)
        #loss_fn = SimplifiedContrastiveLoss(margin=1.0)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        best_val_roc = 0.0
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = 0.0
            train_steps = 0

            for x1, x2, y in train_dataset:
                with tf.GradientTape() as tape:
                    preds = self.teacher_model([x1, x2], training=True)
                    loss = loss_fn(y, preds)
                    
                    if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                        continue

                grads = tape.gradient(loss, self.teacher_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.teacher_model.trainable_variables))

                train_loss += float(loss.numpy())
                train_steps += 1

            if train_steps == 0:
                continue

            avg_train_loss = train_loss / train_steps

            val_loss_sum = 0.0
            val_steps = 0
            for x1, x2, y in val_dataset:
                preds = self.teacher_model([x1, x2], training=False)
                loss = loss_fn(y, preds)
                if not (tf.math.is_nan(loss) or tf.math.is_inf(loss)):
                    val_loss_sum += float(loss.numpy())
                    val_steps += 1
            
            if val_steps == 0:
                continue
                
            avg_val_loss = val_loss_sum / val_steps

            val_metrics = compute_metrics(self.teacher_model, val_dataset, self.device_name)

            print(f" 教师训练损失: {avg_train_loss:.4f}, 验证ROC: {val_metrics['roc_auc']:.4f}")
            
            if val_metrics['roc_auc'] > best_val_roc:
                best_val_roc = val_metrics['roc_auc']
                self.teacher_model.save_weights(os.path.join(self.outpath, "teacher_best.ckpt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("教师模型早停触发!")
                    break

        self.teacher_model.load_weights(os.path.join(self.outpath, "teacher_best.ckpt"))
        print(f"? 教师模型训练完成! 最佳ROC-AUC: {best_val_roc:.4f}\n")
    # ---------------- Evaluate ----------------
    def evaluate_final(self):
        """最终评估：只评估 stage1 与 stage2（stage2 为最终模型）"""
        print("\n" + "=" * 60)
        print("最终评估（两阶段训练）")
        print("=" * 60)

        test_dataset_raw = IndexedNPZHiCDataset(
            self.raw_npz, self.test_idx_raw, self.batch_size, add_noise=False
        )

        results = {}
        stages = [
            ('stage1', "student_stage1_best.ckpt", "阶段1-预训练"),
            ('stage2', "student_stage2_best.ckpt", "阶段2-蒸馏（最终）")
        ]

        for stage_key, ckpt_file, stage_name in stages:
            print(f"\n{stage_name}（原始测试集）:")
            ckpt_path = os.path.join(self.outpath, ckpt_file)
            if os.path.exists(ckpt_path + '.index'):
                self.student_model.load_weights(ckpt_path)
                metrics = compute_metrics(self.student_model, test_dataset_raw, self.device_name)
                results[stage_key] = metrics
                self._print_metrics(metrics)
            else:
                print(f" 未找到权重文件: {ckpt_path}.index")

        # improvement Stage2 vs Stage1
        if 'stage1' in results and 'stage2' in results:
            print("\n性能改进（Stage2 vs Stage1）:")
            for metric in ['roc_auc', 'pr_auc', 'f1_score']:
                diff = results['stage2'][metric] - results['stage1'][metric]
                pct = diff / (results['stage1'][metric] + 1e-12) * 100
                print(f" {metric}: {diff:+.4f} ({pct:+.1f}%)")

        self._plot_comparison(results)
        self._save_report(results)
        return results

    def _print_metrics(self, metrics):
        for k, v in metrics.items():
            if k not in ['scores', 'true_labels']:
                print(f" {k}: {v:.4f}")

    def _plot_comparison(self, results):
        """两阶段对比图（2x2）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (1) bar metrics
        ax = axes[0, 0]
        metrics_names = ['ROC-AUC', 'PR-AUC', 'F1-Score']
        stage_keys = ['stage1', 'stage2']
        stage_names = ['Stage1', 'Stage2(final)']

        x = np.arange(len(metrics_names))  # x: [0, 1, 2, 3]
        width = 0.35  # Bar width

        # Ensure that the length of values is the same as the number of metrics
        for i, (stage_key, stage_name) in enumerate(zip(stage_keys, stage_names)):
            if stage_key in results:
                # Get the metrics values for the current stage
                values = [
                    results[stage_key]['roc_auc'],
                    results[stage_key]['pr_auc'],
                    results[stage_key]['f1_score'],
                    # results[stage_key]['separation_index'] # Unused metric (commented out)
                ]

                # Ensure values has the same length as metrics_names
                if len(values) != len(metrics_names):
                    raise ValueError(
                        f"Length of values ({len(values)}) does not match the length of metrics_names ({len(metrics_names)})")

                # Plot bars with the offset applied based on i
                ax.bar(x + i * width, values, width, label=stage_name)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison (2 Stages)')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) training loss
        ax = axes[0, 1]
        for stage in ['stage1', 'stage2']:
            if self.history[stage]['train_loss']:
                ax.plot(self.history[stage]['train_loss'], label=f'{stage}-train_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (3) ROC curves
        ax = axes[1, 0]
        for stage_key, stage_name in zip(stage_keys, stage_names):
            if stage_key in results:
                fpr, tpr, _ = roc_curve(results[stage_key]['true_labels'], results[stage_key]['scores'])
                ax.plot(fpr, tpr, label=f'{stage_name} (AUC={results[stage_key]["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (4) score distribution for final model
        ax = axes[1, 1]
        if 'stage2' in results:
            scores_pos = results['stage2']['scores'][results['stage2']['true_labels'] == 1]
            scores_neg = results['stage2']['scores'][results['stage2']['true_labels'] == 0]
            ax.hist(scores_neg, bins=30, alpha=0.5, label='Similar(0)', density=True)
            ax.hist(scores_pos, bins=30, alpha=0.5, label='Different(1)', density=True)
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            ax.set_title('Final(Stage2) Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Two-Stage Training Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        out_png = os.path.join(self.outpath, 'two_stage_comparison.png')
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n对比图已保存至: {out_png}")

    def _save_report(self, results):
        """保存两阶段报告"""
        report_file = os.path.join(self.outpath, 'two_stage_report.txt')
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("两阶段训练实验报告（Stage1预训练 + Stage2蒸馏）\n")
            f.write("=" * 70 + "\n\n")

            stage_info = [
                ('stage1', "阶段1 - 预训练"),
                ('stage2', "阶段2 - 蒸馏（最终）")
            ]

            for stage_key, stage_name in stage_info:
                if stage_key in results:
                    f.write(f"\n{stage_name}:\n")
                    f.write("-" * 40 + "\n")
                    for k, v in results[stage_key].items():
                        if k not in ['scores', 'true_labels']:
                            f.write(f" {k}: {v:.4f}\n")

            if 'stage1' in results and 'stage2' in results:
                f.write("\n" + "=" * 40 + "\n")
                f.write("性能改进分析（Stage2 vs Stage1）:\n")
                f.write("-" * 40 + "\n")
                for metric in ['roc_auc', 'pr_auc', 'f1_score']:
                    improvement = results['stage2'][metric] - results['stage1'][metric]
                    improvement_pct = improvement / (results['stage1'][metric] + 1e-12) * 100
                    f.write(f" {metric}: {improvement:+.4f} ({improvement_pct:+.1f}%)\n")

        print(f"报告已保存至: {report_file}")
        
        
# ==================== 工具函数区域 ====================

def get_curated_indices_safe(raw_idx_list, raw_data, cur_data):
    """
    安全的数据映射函数：通过 (染色体:坐标) 进行精确匹配
    确保精选数据是原始数据对应位置的严格子集
    """
    # 1. 获取原始数据的所有坐标
    raw_coords_all = raw_data['coords']
    
    # 2. 找出当前 split (raw_idx_list) 对应的那些坐标集合
    #    注意：这里假设 coords 是字符串数组，如 ['chr1:0-5000', ...]
    selected_raw_coords = set(raw_coords_all[raw_idx_list])
    
    # 3. 遍历精选数据的所有坐标，只有当它存在于 selected_raw_coords 中时才保留
    cur_coords_all = cur_data['coords']
    curated_indices = []
    
    for i, coord in enumerate(cur_coords_all):
        if coord in selected_raw_coords:
            curated_indices.append(i)
            
    return np.array(curated_indices)

# ==================== End 工具函数 ====================

def main():
    # 配置参数
    '''
    raw_npz = "../IMR90-hECS/IMR90_hESC_datas.npz"
    curated_npz = "../IMR90-hECS/IMR90_hESC_datas_cleaned_balance.npz"
    splits_path = "./ave_datas/IMR90-hECS/ave_balanced_5fold_splits-1.npz"
    outpath_base = "./ave_datas/IMR90-hECS/model-8"
    '''
    
    raw_npz = "../K562-GM12878(25G)/output_celue4-1_5kb-2.npz"
    curated_npz = "../K562-GM12878(25G)/output_celue4-1_5kb-2_cleaned_balance-1.npz"
    splits_path = "./ave_datas/K562-GM12878/ave_balanced_5fold_splits-1.npz"
    outpath_base = "./ave_datas/Debug/model-fix"
    
    '''
    raw_npz = "../IMR90-hECS/IMR90_hESC_datas.npz"
    curated_npz = "../IMR90-hECS/IMR90_hESC_datas_cleaned_balance.npz"
    splits_path = "./ave_datas/K562-GM12878/ave_balanced_5fold_splits-1.npz"
    outpath_base = "./ave_datas/cross_cell_lines/model-fix(IMR90-hESC)-original"
    '''
    
    # 模型参数
    embed_dim = 16
    num_heads = 2
    bin_size = 5000
    dropout_rate = 0.3
    batch_size = 32
    
    # 设置随机种子
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    device_name = "/GPU:0"
    
    print("\n" + "="*70)
    print("Hi-C差异分析 - 两阶段训练策略（五折交叉验证，使用预定义划分）")
    print("="*70)
    
    # 加载预定义划分
    splits, n_folds = load_cv_splits(splits_path)
    
    all_metrics_names = ['roc_auc', 'pr_auc', 'accuracy', 'precision',
                         'recall', 'f1_score']
   
    
    # 总体汇总文件
    summary_file = os.path.join(outpath_base, 'cv_summary.txt')
    os.makedirs(outpath_base, exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write("Fold,Train_Samples,Val_Samples,Test_Samples,"
                "ROC_AUC,PR_AUC,Accuracy,Precision,Recall,F1,MeanPerf,SepIndex\n")
    
    fold_results = []
    
    target_fold = 3
    
    for fold in [target_fold]:
        print(f'\n{"#" * 60}')
        print(f'# Fold {fold + 1}/{n_folds}')
        print(f'{"#" * 60}\n')
        
        outpath = os.path.join(outpath_base, f'fold_{fold+1}')
        os.makedirs(outpath, exist_ok=True)
        
        # 获取索引（原始和精选数据使用相同的划分索引，因为染色体一致）
        train_idx = splits[fold]['train_idx']
        val_idx = splits[fold]['val_idx']
        test_idx = splits[fold]['test_idx']
        
        # 由于人工精选数据与原始数据染色体一致，使用相同索引
        train_idx_raw = train_idx
        val_idx_raw = val_idx
        test_idx_raw = test_idx
        '''
        # 在每个 fold 开始时加载数据并映射索引，确保染色体相同
        raw_data = np.load(raw_npz, allow_pickle=True)
        cur_data = np.load(curated_npz, allow_pickle=True)
        raw_coords = np.array([c.split(':')[0] for c in raw_data['coords']])
        cur_coords = np.array([c.split(':')[0] for c in cur_data['coords']])
        
        # 根据染色体名称映射索引（关键！）
        def get_curated_indices_from_raw_indices(raw_indices):
            raw_chroms = raw_coords[raw_indices]
            curated_indices = []
            for chrom in np.unique(raw_chroms):
                raw_mask = raw_chroms == chrom
                cur_mask = cur_coords == chrom
                if not np.any(cur_mask):
                    continue
                curated_indices.extend(np.where(cur_mask)[0])
            return np.array(curated_indices)
            
        # 使用映射函数获取 curated 的索引，确保染色体相同
        train_idx_cur = get_curated_indices_from_raw_indices(train_idx_raw)
        val_idx_cur = get_curated_indices_from_raw_indices(val_idx_raw)
        test_idx_cur = get_curated_indices_from_raw_indices(test_idx_raw)
        '''
        print("正在进行安全索引映射...")
        raw_data_obj = np.load(raw_npz, allow_pickle=True)
        cur_data_obj = np.load(curated_npz, allow_pickle=True)
            
        # 使用新的安全函数进行精确坐标匹配
        train_idx_cur = get_curated_indices_safe(train_idx_raw, raw_data_obj, cur_data_obj)
        val_idx_cur = get_curated_indices_safe(val_idx_raw, raw_data_obj, cur_data_obj)
        test_idx_cur = get_curated_indices_safe(test_idx_raw, raw_data_obj, cur_data_obj)

        # 释放内存 (可选，如果内存紧张)
        del raw_data_obj
        del cur_data_obj
        import gc
        gc.collect()
        
        print(f"训练样本 - 原始: {len(train_idx_raw)}, 精选: {len(train_idx_cur)}")
        print(f"验证样本 - 原始: {len(val_idx_raw)}, 精选: {len(val_idx_cur)}")
        print(f"测试样本 - 原始: {len(test_idx_raw)}, 精选: {len(test_idx_cur)}\n")
        
        # 构建模型
        data_sample = np.load(raw_npz, allow_pickle=True)
        input_shape = (*data_sample['X1'].shape[1:], 1)
        
        teacher_model = SimplifiedHiCNet(
            input_shape=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bin_size=bin_size,
            dropout_rate=dropout_rate
        )
        teacher_model.build([(None, *input_shape), (None, *input_shape)])
        
        student_model = SimplifiedHiCNet(
            input_shape=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bin_size=bin_size,
            dropout_rate=dropout_rate
        )
        student_model.build([(None, *input_shape), (None, *input_shape)])
        
        # 创建两阶段训练器
        trainer = TwoStageTrainer(
            teacher_model, student_model,
            raw_npz, curated_npz,
            train_idx_raw, val_idx_raw, test_idx_raw,
            train_idx_cur, val_idx_cur, test_idx_cur,
            batch_size=batch_size,
            device_name=device_name,
            outpath=outpath
        )
        
        # 执行两阶段训练
        stage1_result = trainer.stage1_pretrain(epochs=100, learning_rate=0.0003)

        # ==================== 方案B：alpha 自适应（整体降低）====================
        ratio = len(train_idx_cur) / (len(train_idx_raw) + 1e-8)

        # 整体偏小，避免 teacher 把 student 拉偏导致 0.5 阈值下指标崩
        initial_alpha = float(np.clip(0.05 + 0.25 * ratio, 0.05, 0.30))
        final_alpha = float(np.clip(0.02 + 0.10 * ratio, 0.02, 0.15))

        print(f"[Alpha自适应] ratio={ratio:.4f}, initial_alpha={initial_alpha:.3f}, final_alpha={final_alpha:.3f}")
        # ==================== end 方案B ====================

        stage2_result = trainer.stage2_distillation(
            epochs=100,
            learning_rate=0.0002,
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            temperature=1.5
        )

        # 最终评估（stage2即为最终模型）
        results = trainer.evaluate_final()
        fold_results.append(results['stage2'])  # 收集最终学生模型结果
        
        # 记录到汇总文件
        with open(summary_file, 'a') as f:
            f.write(f"{fold + 1},{len(train_idx)},{len(val_idx)},{len(test_idx)},"
                    f"{results['stage2']['roc_auc']:.4f},{results['stage2']['pr_auc']:.4f},"
                    f"{results['stage2']['accuracy']:.4f},{results['stage2']['precision']:.4f},"
                    f"{results['stage2']['recall']:.4f},{results['stage2']['f1_score']:.4f},")
    
    # 汇总所有fold结果
    print("\n" + "=" * 60)
    print("5折交叉验证汇总结果")
    print("=" * 60 + "\n")
    
    # 计算均值和标准差
    summary_stats = {}
    for metric in all_metrics_names:
        #values = [fold_results[i][metric] for i in range(n_folds)]
        # 将 n_folds 改为 len(fold_results)
        values = [fold_results[i][metric] for i in range(len(fold_results))]
        summary_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    # 打印汇总
    print(f"{'指标':<20} {'均值':>10} {'标准差':>10} {'各Fold结果'}")
    print("-" * 75)
    for metric in all_metrics_names:
        stats = summary_stats[metric]
        values_str = ', '.join([f'{v:.4f}' for v in stats['values']])
        print(f"{metric:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} [{values_str}]")
    
    # 保存汇总结果
    with open(summary_file, 'a') as f:
        f.write(f"\n\n# ========== Summary Statistics ==========\n")
        f.write(f"Metric,Mean,Std,Fold1,Fold2,Fold3,Fold4,Fold5\n")
        for metric in all_metrics_names:
            stats = summary_stats[metric]
            values_str = ','.join([f'{v:.4f}' for v in stats['values']])
            f.write(f"{metric},{stats['mean']:.4f},{stats['std']:.4f},{values_str}\n")
    
    print("\n" + "="*70)
    print("实验完成！两阶段训练策略。")
    print(f"所有结果已保存至: {outpath_base}")
    print("="*70)
    
if __name__ == "__main__":
    main()
