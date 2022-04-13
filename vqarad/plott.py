import pandas as pd
import matplotlib.pyplot as plt
def my_plot(df):

    # print(epochs)
    # print()
    colorr= ["green","green","blue","blue","red","red"]
    linewid= [6,5,4,3,2,1]



# importing package
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # # create data
    # x = [1,2,3,4,5]
    # y = [3,3,3,3,3]
    
    # # plot lines
    # plt.plot(x, y, label = "line 1")
    # plt.plot(y, x, label = "line 2")
    # plt.plot(x, np.sin(x), label = "curve 1")
    # plt.plot(x, np.cos(x), label = "curve 2")
    # plt.legend()
    # plt.show()
    j=0
    # res = ini_list.strip('][').split(', ')
    # df["overalla_ccuracy"]= pd.to_numeric(df["overalla_ccuracy"])
    # print(type(df["overalla_ccuracy"]))
    # df["loss"]= pd.to_numeric(df["loss"])
    for i in range(len(df)):
        
        epochs = df.loc[i,"epoch"]
        epoo = [epoch+1 for epoch in range(epochs)] 
        # print(epoo) 
        losses=df.loc[i,"loss"].strip('][').split(', ')
        accs = df.loc[i,"overalla_ccuracy"].strip('][').split(', ')
        # print(type(float(losses[0])))
        losses=[float(loss) for loss in losses]
        accs = [float(acc) for acc in accs]
        # df["loss"]= pd.to_numeric(df["loss"])
        # print("df.loc", df.loc[i,"loss"])
        stringlable1=str(df.loc[i,"model_name"])+ "_loss_plot"
        stringlabel2=str(df.loc[i, "model_name"]) + "_acc_plot"
        plt.plot(epoo, losses, color=colorr[j], label=stringlable1,linewidth=linewid[j])
        j+=1
        plt.plot(epoo, accs, color=colorr[j],\
        label=stringlabel2)
        j+=1
        plt.legend()
    plt.legend()
    plt.show()
    plt.savefig('comparemodel.png')

    # plt.plot(df["epoch"], loss, color="blue")
    # plt.plot(epochs, acc, color="red")
    # plt.show()
    # plt.savefig('train.png')
df = pd.read_excel("output_train.xlsx")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
my_plot(df)


# import matplotlib.pyplot as plt

# # Plot a simple line chart
# plt.plot(df['x'], df['y'], color='g', label='Line y')

# # Plot another line on the same chart/graph
# plt.plot(df['x'], df['z'], color='r', label='Line z', marker='o')

# plt.legend()
# plt.show()