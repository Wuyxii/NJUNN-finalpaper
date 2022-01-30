压缩包中包含：
1. 配置信息 env.sh
2. 训练脚本 train.sh 
    传入参数为根目录，即包含训练数据的文件
    如 ./train.sh ../input/sequence
3. 测试脚本 test.sh 
    传入参数依次为预测F的模型参数，预测U的模型参数，预测Structure的模型参数
    如 ./test.sh f_model.param u_model.param structure_model.param 20pred.csv
4. 训练代码 train.py
    训练完后会保存三个模型参数
5. 测试代码
    加载训练数据后预测结果，并对训练模型进行画图分析 test.py
6. 训练好的三个模型 f_model.param, u_model.param, structure_model.param
7. 预测结果 F_pred.csv, U_pred.csv, Stru_pred.csv
8. 实验报告
