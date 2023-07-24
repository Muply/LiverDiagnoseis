import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from mymodel_newdata import myModel
from dataset_newdata import MyDataset
import numpy as np

train_history = "./train_history/model_doubleConv_train_newdata_train_history.txt"
device = torch.device("cuda:0")

epoch = 150
train_txt_path = './train_cfgs/train.txt'
valid_txt_path = './train_cfgs/test.txt'
batch_size = 128
nw = 8
save_path = './train_history/model_doubleConv_newdata_train.pth'
save_path_fin = save_path[:-4] + '_fin.pth'

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_transform = {
    "train": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.43304095, 0.29830533, 0.18766068), (0.2689185, 0.22025496, 0.15420981))]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.43304095, 0.29830533, 0.18766068), (0.2689185, 0.22025496, 0.15420981))])}

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=data_transform["train"])
valid_data = MyDataset(txt_path=valid_txt_path, transform=data_transform["val"])
train_num = len(train_data)
val_num = len(valid_data)

# 构建DataLoder
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nw)
validate_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=nw)

# 定义网络
net = myModel()
# net.load_state_dict(torch.load('./train_history/model_doubleConv_newdata_train_1_50_50_1.pth'))
net.to(device)

# 损失函数 优化器
wit = torch.tensor([0.50, 0.50], dtype=torch.float32).cuda(device)
loss_function = nn.CrossEntropyLoss(weight=wit)
#loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
best_acc = 0.0

# 训练
for epoch in range(epoch):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    scheduler.step()
    print()
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            # print(predict_y)
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        with open(train_history, "a") as f:
            f.writelines('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' % (epoch + 1, running_loss / step, val_accurate))

torch.save(net.state_dict(), save_path_fin)
print('Best acc:' + str(best_acc))
with open(train_history, "a") as f:
    f.writelines('Best acc:' + str(best_acc) + '\n')
print('Finished Training')
