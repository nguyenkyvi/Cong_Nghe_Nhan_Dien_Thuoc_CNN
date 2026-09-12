import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv
import torch.nn.functional as F

# --- 1. THIẾT LẬP THIẾT BỊ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '.' 

# Transform cơ bản, không có RandomRotation hay RandomFlip (Gây ra biểu đồ giật)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform)
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

# --- 2. CẤU HÌNH RESNET-50 BẢN GỐC ---
# ResNet-50 học từ đầu (weights=None)
model = models.resnet50(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5), 
    nn.Linear(num_ftrs, len(class_names))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Learning rate 0.0005 (Hơi cao so với weights=None nên gây ra spike ở Epoch 10)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 30 # Chạy đủ 30 Epochs
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
best_acc = 0.0

print(f"\n[ĐANG CHẠY] Huấn luyện ResNet-50 bản gốc (Sẽ có dao động mạnh)...")

for epoch in range(num_epochs):
    for phase in ['train', 'valid']:
        if phase == 'train': model.train()
        else: model.eval()
        
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
        epoch_loss = running_loss / dataset_sizes[phase]

        if phase == 'train':
            history['train_acc'].append(epoch_acc)
            history['train_loss'].append(epoch_loss)
        else:
            history['val_acc'].append(epoch_acc)
            history['val_loss'].append(epoch_loss)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')

    print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {history["train_acc"][-1]:.4f} | Val Acc: {history["val_acc"][-1]:.4f}')

# --- 3. VẼ BIỂU ĐỒ (TÊN TIẾNG VIỆT NHƯ TRONG HÌNH) ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Độ chính xác thực tế')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Sai số thực tế')
plt.legend()

plt.savefig('bieu_do_CNN_real.png')

# --- 4. XUẤT CSV ---
test_dir = os.path.join(data_dir, 'test')
if os.path.exists(test_dir):
    with open('ket_qua_phan_tram_chi_tiet.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Ten File Anh', 'Ket Qua Du Doan', 'Do Tu Tin (%)'])
        model.eval()
        for root, dirs, files in os.walk(test_dir):
            for ten_file in files:
                if ten_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, ten_file)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_t = transform(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = model(img_t)
                            probs = F.softmax(out, dim=1)[0] * 100
                            conf, pred = torch.max(probs, 0)
                            writer.writerow([ten_file, class_names[pred.item()], f"{conf.item():.2f}%"])
                    except: continue
    print("\n[HOÀN TẤT] Đã sinh lại biểu đồ gốc của Vĩ!")