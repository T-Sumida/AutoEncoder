#coding:utf-8

from AutoEncorder import AutoEncorder,Activation
import numpy as np
if __name__ == "__main__":
    # AutoEncorderクラスのインスタンスを生成
    auto = AutoEncorder(6,2)
    
    # 乱数データを作成
    data = np.random.rand(36).reshape(6,6)

    # 乱数データをフィッティング
    auto._fitting(data)

    # 乱数データをエンコード
    sample = np.random.rand(6)
    hidden,output = auto._encode(sample)

    print("hiden:",hidden)
    print("error:",sample-output)



    
