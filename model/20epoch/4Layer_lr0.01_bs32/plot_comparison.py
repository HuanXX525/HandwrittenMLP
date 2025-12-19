import matplotlib.pyplot as plt
import re

def parse_log(file_path):
    epochs = []
    losses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = re.findall(r'Epoch (\d+) \| Total Avg Loss: ([\d.]+)', content)
        for epoch, loss in matches:
            epochs.append(int(epoch))
            losses.append(float(loss))
    return epochs, losses

adam_epochs, adam_losses = parse_log('Adam.log')
momentum_epochs, momentum_losses = parse_log('Momentum.log')
sgd_epochs, sgd_losses = parse_log('SGD.log')

plt.figure(figsize=(10, 6))
plt.plot(adam_epochs, adam_losses, 'b-', label='Adam', linewidth=2, marker='o', markersize=4)
plt.plot(momentum_epochs, momentum_losses, 'r-', label='Momentum', linewidth=2, marker='s', markersize=4)
plt.plot(sgd_epochs, sgd_losses, 'g-', label='SGD', linewidth=2, marker='^', markersize=4)

plt.xlabel('(Epoch)', fontsize=12)
plt.ylabel('(Loss)', fontsize=12)
plt.title('Compare', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 optimizer_comparison.png")
plt.show()
