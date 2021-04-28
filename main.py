from libsvm.svmutil import *
from libsvm.svm import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def list_of_dict_to_list_of_list(list_of_dict: list) -> list:
    """
    将元素为字典的列表转换成元素为列表的列表，仅取value

    Args:
        list_of_dict (list): 元素为字典的列表

    Returns:
        list: 元素为列表的列表
    """
    list_of_list = [[v for v in d.values()] for d in list_of_dict]
    return list_of_list


def get_dividing_point(y: list):
    """
    找出不同样例的分界点

    Args:
        y (list): 数据标签

    Returns:
        int: -1表示全部相同，否则表示分界点

    """
    last = y[0]
    for i, yi in enumerate(y):
        if yi != last:
            return i
        else:
            last = yi
    return -1


def scatter_training_set(x: list, y: list, axes):
    """
    绘制训练集散点图

    Args:
        x (list): 数据特征
        y (list): 数据标签
        axes (matplotlib.axes._base._AxesBase): 要绘图的Axes实例

    Returns:
        None
    """
    x_array = np.array(list_of_dict_to_list_of_list(x))
    x1 = x_array[:, 0]
    x2 = x_array[:, 1]
    dividing_point = get_dividing_point(y)
    axes.scatter(x1[:dividing_point], x2[:dividing_point])
    axes.scatter(x1[dividing_point:], x2[dividing_point:])


def leave_one_out(x: list, y: list, param_str: str):
    """
    进行留一交叉验证

    Args:
        x (list): 数据特征
        y (list): 数据标签
        param_str (str): SVM参数指令

    Returns:
        留一交叉验证精度
    """
    param_str += " -v " + str(len(y))
    accuracy = svm_train(y, x, param_str)
    return accuracy


def solve_predict(x: list, y: list, param_str: str):
    """
    训练模型SVM并用于分类

    Args:
        x (list): 数据特征
        y (list): 数据标签
        param_str (str): SVM参数指令

    Returns:
        p_label, p_acc, p_val, model
    """
    prob = svm_problem(y, x)
    param = svm_parameter(param_str)
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y, x, model)
    return p_label, p_acc, p_val, model


def tuning_gauss(x: list, y: list, c_range: np.ndarray, g_range: np.ndarray):
    """
    SVM高斯核调参

    Args:
        x (list): 数据特征
        y (list): 数据标签
        c_range (np.ndarray): c参数所有取值
        g_range (np.ndarray): g参数所有取值

    Returns:
        best_result (dict): 调参的最优结果，包含精度和c、g取值
        result_frame (pd.DataFrame): 调参过程中所有c、g和对应精度
    """
    best_result = {"Accuracy": -1, "c": -1, "g": -1}
    result_file_name = "best_result.txt"
    result_array = []
    clear_file(result_file_name)
    for c in c_range:
        for g in g_range:
            param_str = '-q -t 2 -c ' + str(c) + ' -g ' + str(g)
            accuracy = leave_one_out(x, y, param_str)
            result_array.append([float(format(c, '.6f')), float(format(g, '.6f')), accuracy])
            if accuracy >= best_result["Accuracy"]:
                best_result["Accuracy"] = accuracy
                best_result["c"] = c
                best_result["g"] = g
                append_dict_to_file(result_file_name, best_result)
    result_frame = pd.DataFrame(result_array, columns=['c', 'g', 'Accuracy'])
    return best_result, result_frame


def clear_file(filename: str):
    """
    清空文件

    Args:
        filename (str): 文件名

    Returns:
        None
    """
    with open(filename, mode='r+', encoding='UTF-8') as file_object:
        file_object.truncate()


def append_dict_to_file(filename: str, content: dict):
    """
    将字典内容写入文件

    Args:
        filename (str): 文件名
        content (dict): 要写入的字典

    Returns:
        None
    """
    newline = ''  # 要写入的内容
    for k, v in content.items():
        newline += str(k) + ': ' + str(v) + '\t'
    newline += '\n'
    append_to_file(filename, newline)


def append_to_file(filename: str, content: str):
    """
    将字符串写入文件

    Args:
        filename (str): 文件名
        content (str): 要写入的字符串

    Returns:
        None
    """
    with open(filename, mode='r+', encoding='UTF-8') as file_object:
        file_object.seek(0, 2)
        file_object.writelines(content)


def plot_tuning_result(result_frame: pd.DataFrame):
    """
    绘制调参结果的热力图

    Args:
        result_frame (pd.DataFrame): 调参结果

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # sns.set()
    result_frame = result_frame.pivot("c", "g", "Accuracy")
    hm = sns.heatmap(result_frame, ax=ax, cmap="YlGnBu")
    hm.set_xlabel(hm.get_xlabel(), labelpad=0, rotation=0)
    plt.yticks(rotation=0)
    plt.savefig('parameter heat map.png', dpi=260)


def calculate_laplace_kernel(x: list, y: list, gamma: float, result_file_name: str):
    """
    计算拉普拉斯核并写入文件

    Args:
        x (list): 数据特征
        y (list): 数据标签
        gamma (float): gamma参数
        result_file_name (str): 要写入的文件名

    Returns:
        None
    """
    x_array = np.array(list_of_dict_to_list_of_list(x))
    clear_file(result_file_name)
    for i in range(len(y)):
        kernels = []
        for j in range(len(y)):
            x_dif = x_array[i, :] - x_array[j, :]  # 可以利用传播性质优化计算
            x_distance = np.power(np.sum(np.power(x_dif, 2)), 0.5)
            kernel = np.exp(-gamma * x_distance)
            kernels.append(kernel)
        content = str(y[i]) + " 0:" + str(i + 1)
        for k in range(len(y)):
            content += ' ' + str(k + 1) + ':' + str(kernels[k])
        content += '\n'
        append_to_file(result_file_name, content)


def use_laplace(x, y, c: float):
    """
    使用拉普拉斯核进行留一交叉验证和分类

    Args:
        x (list): 数据特征
        y (list): 数据标签
        c (float): c参数

    Returns:
        accuracy, p_label, p_acc, p_val, model
    """
    param_str = '-q -t 4 -c ' + str(c)
    accuracy = leave_one_out(x, y, param_str)
    p_label, p_acc, p_val, model = solve_predict(x, y, param_str)
    return accuracy, p_label, p_acc, p_val, model


def tuning_laplace(x: list, y: list, kernel_file_name: str, c_range: np.ndarray, g_range: np.ndarray):
    """
    SVM拉普拉斯核调参

    Args:
        x (list): 数据特征
        y (list): 数据标签
        kernel_file_name (str): 要写入拉普拉斯核的文件名
        c_range (np.ndarray): c参数所有取值
        g_range (np.ndarray): g参数所有取值

    Returns:
        best_result (dict): 调参的最优结果，包含精度和c、g取值
        result_frame (pd.DataFrame): 调参过程中所有c、g和对应精度
    """
    best_result = {"Accuracy": -1, "c": -1, "g": -1}
    result_file_name = "best_laplace_result.txt"
    result_array = []
    clear_file(result_file_name)
    for g in g_range:
        calculate_laplace_kernel(x, y, g, kernel_file_name)
        ly, lx = svm_read_problem(kernel_file_name)
        for c in c_range:
            param_str = '-q -t 4 -c ' + str(c)
            accuracy = leave_one_out(lx, ly, param_str)
            result_array.append([float(format(c, '.2f')), float(format(g, '.2f')), accuracy])
            # result_array.append([c, g, accuracy])
            if accuracy >= best_result["Accuracy"]:
                best_result["Accuracy"] = accuracy
                best_result["c"] = c
                best_result["g"] = g
                append_dict_to_file(result_file_name, best_result)
    result_frame = pd.DataFrame(result_array, columns=['c', 'g', 'Accuracy'])
    return best_result, result_frame


def plot_sv(model, customed_model: bool, axes, x: np.ndarray = np.array([])):
    """
    在图中标注支持向量

    Args:
        model (): SVM模型
        customed_model (bool): 是否使用自定义核（拉普拉斯核）
        axes (matplotlib.axes._base._AxesBase): 要绘图的Axes实例
        x (np.ndarray): 使用自定义核时的原始数据

    Returns:
        None
    """
    if not customed_model:
        sv_dict = model.get_SV()
        sv = np.array(list_of_dict_to_list_of_list(sv_dict))
    else:
        if x.size == 0:
            raise Exception("x数据缺失")
        sv_indices = np.array(model.get_sv_indices(), dtype=np.int32) - 1
        sv = x[sv_indices]
    x1 = sv[:, 0]
    x2 = sv[:, 1]
    plt.scatter(x1, x2, marker='o', facecolor='none', edgecolors='black', s=200)


def plot_data_and_sv(x, y, model, customed_model: bool, title: str, fig_file_name: str = "data and SV"):
    """
    绘制原始数据并标注支持向量

    Args:
        x (list): 数据特征
        y (list): 数据标签
        model (): SVM模型
        customed_model (bool): 是否使用自定义核（拉普拉斯核）
        title (str): 绘图标题
        fig_file_name (str): 保存图片的文件名

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1)
    scatter_training_set(x, y, axes)
    x = np.array(list_of_dict_to_list_of_list(x))
    plot_sv(model=model, customed_model=customed_model, axes=axes, x=x)
    props = {'xlabel': '密度', 'ylabel': '含糖率', 'title': title}
    axes.set(**props)
    axes.set_ylabel(axes.get_ylabel(), labelpad=20, rotation=0)
    plt.savefig(fig_file_name + ".png", dpi=260)


if __name__ == '__main__':
    param_str = '-q -t 2 -c 1.4 -g 110'
    y, x = svm_read_problem('training set.txt')
    '''高斯'''
    accuracy = leave_one_out(x, y, param_str)
    p_label, p_acc, p_val, gauss_model = solve_predict(x, y, param_str)
    plot_data_and_sv(x=x, y=y, model=gauss_model, customed_model=False, title="SVM-高斯核, C=1.4, γ=110",
                     fig_file_name="gauss data and SV")
    # best_gauss_result, gauss_result_frame = tuning_gauss(x, y, np.linspace(1, 10, int((10 - 1) * 1) + 1),
    #                                                      np.linspace(0, 128, int((128 - 0) * 1) + 1))
    # best_gauss_result, gauss_result_frame = tuning_gauss(x, y, np.logspace(-4, 4, num=513, base=10),
    #                                                      np.logspace(-4, 4, num=513, base=10))
    # plot_tuning_result(gauss_result_frame)
    '''拉普拉斯'''
    calculate_laplace_kernel(x, y, gamma=9, result_file_name="laplace_kernel.txt")
    ly, lx = svm_read_problem("laplace_kernel.txt")
    l_accuracy, l_p_label, l_p_acc, l_p_val, laplace_model = use_laplace(lx, ly, c=0.8)
    plot_data_and_sv(x=x, y=y, model=laplace_model, customed_model=True,
                     title="SVM-拉普拉斯核, C=0.8, γ=9",
                     fig_file_name="laplace data and SV")
    # best_laplace_result, laplace_result_frame = tuning_laplace(x, y, "laplace_kernel.txt",
    #                                                            np.logspace(-4, 4, num=129, base=10),
    #                                                            np.logspace(-4, 4, num=129, base=10))
    # best_laplace_result, laplace_result_frame = tuning_laplace(x, y, "laplace_kernel.txt",
    #                                                            np.linspace(0.2, 1.4, int((1.4 - 0.2) * 10) + 1),
    #                                                            np.linspace(0, 40, int((40 - 0) * 10) + 1))
    # plot_tuning_result(laplace_result_frame)
    plt.show()
