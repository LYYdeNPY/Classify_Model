import torch
import torch.nn as nn
import torch.optim as optim
from net import net
from dataloader import train_loader, test_loader

# 定义代价函数
criterion = nn.CrossEntropyLoss()
# 定义优化函数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

print("=============Start Training==============")
correct = 0
total = 0
for epoch in range(20):  # 多次循环遍历数据集
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # 获取输入
        inputs, labels = data

        # 参数梯度置零
        optimizer.zero_grad()

        # 前向+ 反向 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 输出统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i == 2499:  # 10000张图片，2500个batch
            print('[epoch:%d] loss: %.3f     Accuracy on this batch: %.3f%%' %
                  (epoch + 1, running_loss / 2500, 100.0 * correct / total))
            running_loss = 0.0
print("=============Finish Training=============")

# 测试2000张图片
for i in range(3):
    correct = 0
    total = 0
    for data in test_loader:
        inputs, targe = data
        outputs = net(inputs)
        total += targe.size(0)
        _, predicted = torch.max(outputs.data, dim=1)
        correct += (predicted == targe).sum().item()
    print('Accuracy on test set: %.3f %%' % (100 * correct / total))
