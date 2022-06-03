from concurrent.futures import process
from unicodedata import name
import os
from datetime import datetime
from process_data import tranlate_file,random_data,split_data,split_data_for_part

from sklearn.metrics import balanced_accuracy_score,confusion_matrix,accuracy_score
import svmutil as svm



tranlate_file()
# random_data('data_convert','data_random')
# split_data('data_random','train','test')
# split_data_for_part('train','data_train',1)




# path_train=os.path.abspath('data_for_train')
# list_file=os.listdir(path_train)










# if(os.path.exists(os.path.abspath('models'))==False):
#     os.mkdir('models')
# path_model=os.path.abspath('models')

# list_model=os.listdir(path_model)


# # chia train ra 50% để train còn lại để test
# file_50_train = open(path_train+os.sep+"data_train.csv","r")

# tmp= file_50_train.readlines()

# for i in range(0,len(tmp)):
#     if i<=(len(tmp)/2):
#         with open(path_train+os.sep+"train.csv","a") as f:
#             f.write(tmp[i])
#     else:
#         with open(path_train+os.sep+"test.csv","a") as f:
#             f.write(tmp[i])





# #train
# i=0.3
# list_nu= []
# while(i<1):
#     list_nu.append(i)
#     i+=0.3

# list_gamma= []
# for i in range(0,3,1):
#     list_gamma.append(float(i))



# for nu in list_nu:
#     for gamma in list_gamma:
#         # for file_name in list_file:

#             # get time now
#             now = datetime.now()
#             current_time = now.strftime("%H_%M_%S")

#             y,x=svm.svm_read_problem(path_train+os.sep+"data_train.csv")
#             prob=svm.svm_problem(y,x)
#             param=svm.svm_parameter("-s 2 -q -n "+str(float(nu))+" -g "+str(float(gamma)))
#             m=svm.svm_train(prob,param)
#             # remove_csv=file_name.replace(".csv","")
#             svm.svm_save_model(path_model+os.sep+f"data_{nu}_{gamma}_{current_time}.model",m)




# # #test
# # # link to file test 50%
# yt,xt=svm.svm_read_problem(path_train+os.sep+"test.csv")

# f= open(path_train+os.sep+'test.csv')
# arr= f.readlines()

# # get label real 0 1 first
# list_label_real=[]
# for i in arr:
#     tach= i.split(' ')
#     list_label_real.append(tach[0])


# list_label=[]
# list_p_val=[]
# list_ba=[]
# for i in list_model:
#         m=svm.svm_load_model(path_model+"/"+i)
#         p_label, p_acc, p_val = svm.svm_predict(yt, xt, m)
#         list_label.append(os.path.splitext(i)[0])
#         list_p_val.append(p_val)
#         out_label=[]
#         out_p_val=[]
#         for i in list_p_val[0]:
#             out_p_val.append(i[0])
#         for i in list_p_val[0]:
#                 out_label.append(list_label[0])
#         for i in range(0,len(list_p_val)):
#             for j in range(0,len(list_p_val[i])):
#                 if float(list_p_val[i][j][0])>float(out_p_val[j]):
#                         out_p_val[j]=list_p_val[i][j][0]
#                         out_label[j]=list_label[i]
#         acc=balanced_accuracy_score(list_label_real,out_label)
#         list_ba.append(acc)


# print(list_ba)
