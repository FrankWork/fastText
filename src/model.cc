/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    int32_t seed)
    : hidden_(args->dim),
      output_(wo->size(0)),
      grad_(args->dim),
      rng(seed),
      quant_(false) {
  wi_ = wi;
  wo_ = wo;
  args_ = args;
  osz_ = wo->size(0);
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  t_log_.reserve(LOG_TABLE_SIZE + 1);
  initSigmoid();
  initLog();
}

void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi,
                               std::shared_ptr<QMatrix> qwo, bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

// 二分类损失，包括前向和后向 
real Model::binaryLogistic(int32_t target, bool label, real lr) {
  // 内积，sigmoid
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  // Loss 对于 hidden_ 的梯度累加到 grad_ 上
  grad_.addRow(*wo_, target, alpha);
  // Loss 对于 LR 参数的梯度累加到 wo_ 的对应行上
  wo_->addRow(hidden_, target, alpha);
  // 交叉熵
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

// 负采样
real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();

  // 对于正样本和负样本，分别更新 LR
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
    } else {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}

// 层次softmax
real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  // 先确定霍夫曼树上的路径
  const std::vector<bool>& binaryCode = codes[target];
  // 分别对路径上的中间节点做 LR
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    output.mul(*wo_, hidden);
  }
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}

// 普通softmax损失
real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}

// 将输入词（ngram）向量平均，保存到hidden里
void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if(quant_) {
      hidden.addRow(*qwi_, *it);
    } else {
      hidden.addRow(*wi_, *it);
    }
  }
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}

void Model::predict(const std::vector<int32_t>& input, int32_t k, real threshold,
                    std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }
  // 分配k+1个空间
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs) {
    dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, threshold, heap, hidden, output);
  }
  // 因为 heap 中虽然一定是 top-k，但并没有排好序
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(
  const std::vector<int32_t>& input,
  int32_t k,
  real threshold,
  std::vector<std::pair<real, int32_t>>& heap
) {
  predict(input, k, threshold, heap, hidden_, output_);
}

void Model::findKBest(
  int32_t k,
  real threshold,
  std::vector<std::pair<real, int32_t>>& heap,
  Vector& hidden, Vector& output
) const {
  // 计算结果数组
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (output[i] < threshold) continue;
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    // 使用一个堆来保存 topK 的结果，这是算 topK 的标准做法
    heap.push_back(std::make_pair(std_log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Model::dfs(int32_t k, real threshold, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (score < std_log(threshold)) return;
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1) {
    // 只输出叶子节点的结果
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  // 将 score 累加后递归向下收集结果
  real f;
  if (quant_ && args_->qout) {
    f= qwo_->dotRow(hidden, node - osz_);
  } else {
    f= wo_->dotRow(hidden, node - osz_);
  }
  f = 1. / (1 + std::exp(-f));// sigmoid

  dfs(k, threshold, tree[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree[node].right, score + std_log(f), heap, hidden);
}

void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  
  // 将输入词（ngram）向量平均，保存到hidden里
  computeHidden(input, hidden_);
  
  // loss functions, 前向和后向
  if (args_->loss == loss_name::ns) {
    loss_ += negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  // backprop
  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  //将 hidden_ 上的梯度传播到 wi_ 上的对应行
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives_.push_back(i);
    }
  }
  std::shuffle(negatives_.begin(), negatives_.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives_[negpos];
    negpos = (negpos + 1) % negatives_.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  /*构建 Huffman 树的思想：
      1.  找到当前权重最小（出现频率低）的两个子树，将它们合并成为新树，
          用新树来替换原来两个子树，直到只剩下一个树
      2.  哈弗曼树是一种带权路径长度最短的二叉树，可用来构造最优编码
      3.  叶子节点表示词，中间节点表示隐含的词类别
    
    二叉树节点个数：
      n = n0+n1+n2, n0: 度为0的节点，叶子节点
      n0 = n2+1
    最优二叉树：n1 = 0
               n = n0 + 0 + n0-1 = 2*n0 - 1

    算法：
      1. 对输入的叶子节点进行一次排序，复杂度为 O(nlogn) ，
      2. 确定两个下标 leaf 和 node，
         leaf 总是指向当前最小的叶子节点，
         node 总是指向当前最小的非叶子节点，所以，
         最小的两个节点可以从 leaf, leaf - 1, node, node + 1 四个位置中取得
      3. 对每个叶子节点遍历一遍，用2的方法算出节点，总复杂度为 O(n)
      算法整体时间复杂度为 O(nlogn)，空间复杂度为O(1)。
  */
  // 分配所有节点的空间
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }

  // counts 数组保存每个叶子节点的词频，降序排列
  // 叶子节点保存在前`osz_`位置里
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  
  // leaf 指向当前未处理的叶子节点的最后一个，也就是权值最小的叶子节点
  // node 指向当前未处理的非叶子节点的第一个，也是权值最小的非叶子节点
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  
  // 逐个构造所有非叶子节点（i >= osz_, i < 2 * osz - 1)
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    // 当前最小两个节点的下标
    int32_t mini[2];
    // 最小的两个节点可以从 leaf, leaf - 1, node, node + 1 四个位置中取得
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    
    // 更新非叶子节点的属性
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;// 右子树编码为1
  }
  
  // 计算霍夫曼编码
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_); // path to root, 让中间节点从0开始编码
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

void Model::initSigmoid() {
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }
}

void Model::initLog() {
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Model::std_log(real x) const {
  return std::log(x+1e-5);
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

}
