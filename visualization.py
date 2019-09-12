from matplotlib import pyplot as plt

conv_log_path = 'main_conv.log'
deform_conv_log_path = 'main.log'


def load_data(file_path):
    train_acc, test_acc = [], []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if 'accuracy' not in line:
                continue
            line = line.split()
            train_acc.append(float(line[3].rstrip('%,')))
            test_acc.append(float(line[-1].rstrip('%')))
            if len(test_acc) == 70:
                return train_acc, test_acc

    return train_acc, test_acc

def plot_acc():
    pass

def main():
    conv_train_acc, conv_test_acc = load_data(conv_log_path)
    deform_train_acc, deform_test_acc = load_data(deform_conv_log_path)
    epochs = list(range(70))
    plt.figure()
    plt.plot(epochs, conv_train_acc, color='blue', linestyle='-', label='conv_train')
    plt.plot(epochs, conv_test_acc, color='blue', linestyle='--', label='conv_test')
    plt.plot(epochs, deform_train_acc, color='red', linestyle='-', label='deform_conv_train')
    plt.plot(epochs, deform_test_acc, color='red', linestyle='--', label='deform_conv_test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('comparision.png')


if __name__ == '__main__':
    main()