import sys;
import pandas as pd;
import matplotlib.pyplot as plt;
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import math as m;
import tkinter as tk;
from tkinter import filedialog as fd
from scipy.stats import norm
import statistics as st

def my_confusion_matrix(list_1, list_2):
    tn_tmp = tp_tmp = fn_tmp = fp_tmp = 0;
    for i in range(len(list_1)):
        #print("i: ", i,";  list_1[i]: ", list_1[i])
        if list_1[i] == 0 and list_2[i] == 0:
            tn_tmp += 1;
        elif list_1[i] == 0 and list_2[i] == 1:
            fp_tmp += 1;
        elif list_1[i] == 1 and list_2[i] == 0:
            fn_tmp += 1;
        elif list_1[i] == 1 and list_2[i] == 1:
            tp_tmp += 1;
        else:
            print("wrong data in confusion_matrix function!\n")
    return tn_tmp, fp_tmp, fn_tmp, tp_tmp


def my_roc_curve(true_data, score_data):
    #thresholds = [];
    #tmp = 1;
    #while tmp >= 0:
    #    thresholds.append(tmp);
    #    tmp -= 0.05;
    #thresholds.append(0);
    thresholds = score_data.unique();
    fpr = []; tpr = [];
    pred_val = 0;
    auc = 0;
    #print("thr:", thresholds);
    print("thresholds len: ", len(thresholds));
    for i in range(len(thresholds)):
        pred_list = [];
        for j in range(len(score_data)):
            if score_data[j] >= thresholds[i]:
                pred_val = 1;
            else:
                pred_val = 0;
            pred_list.append(pred_val);
        tn, fp, fn, tp = my_confusion_matrix(true_data, pred_list)
        fpr.append(1-(tn/(tn+fp)));
        tpr.append(tp/(fn+tp));
        if len(fpr) > 1:
            auc += (tpr[len(tpr)-1] + tpr[len(tpr)-2]) * abs(fpr[len(fpr)-1] - fpr[len(fpr)-2]) / 2
    print("fpr len:", len(fpr), "tpr len:", len(tpr))
    return fpr, tpr, auc;
      
  
def ROC(true_data, score_data, graphName):
    fpr, tpr, auc = my_roc_curve(true_data, score_data)
    #print("fpr: ", fpr)
    #print("tpr: ", tpr)
    fig = Figure(figsize=(4,4))
    plot1 = fig.add_subplot();
    plot1.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    plot1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot1.set_xlim([0.0, 1.0])
    plot1.set_ylim([0.0, 1.0])
    plot1.set_xlabel('1 - specyficzność')
    plot1.set_ylabel('Czułość')
    plot1.set_title('Krzywa ROC: ' + graphName)
    plot1.legend(loc="lower right")
    
    canvas = FigureCanvasTkAgg(fig, master = window);
    canvas.draw();

    global columnNum;
    canvas.get_tk_widget().grid(column = columnNum, columnspan = 40, row = 2, padx = 5);
    columnNum += 51;
    
    return auc;


def ocen_model_klasyfikacji_binarnej(true_data, pred_data, digits = 3):
    tn, fp, fn, tp = my_confusion_matrix(true_data, pred_data)
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    overall_error_rate = 1 - accuracy
    sensitivity = tp/(fn+tp)
    fnr = fn/(fn+tp)
    specificity = tn/(tn+fp)
    fpr = fp/(tn+fp)
    precision = tp/(fp+tp)
    f1 = (2 * sensitivity * precision) / (sensitivity + precision)

    print('Trafność: ', round(accuracy, digits))
    print('Całkowity współczynnik błędu', round(overall_error_rate, digits))
    print('Czułość: ', round(sensitivity, digits))
    print('Wskaźnik fałszywie negatywnych: ', round(fnr, digits))
    print('Specyficzność: ', round(specificity, digits))
    print('Wskaźnik fałszywie pozytywnych: ', round(fpr, digits))
    print('Precyzja: ', round(precision, digits))
    print('Wynik F1: ', round(f1, digits))
    print('\n');

    wskazniki = [accuracy, sensitivity, specificity, precision, f1];
    return wskazniki;


def klasyfikacja_binarna(path):
    data = pd.read_csv(path);
    
    col = data.columns;
    
    data_1 = data[[col[0],col[1],col[2]]].sort_values(by = col[2], ascending=False, ignore_index=True);
    data_2 = data[[col[0],col[3],col[4]]].sort_values(by = col[4], ascending=False, ignore_index=True);
    
    true_data = data_1[col[0]].map({'>50K': 1, '<=50K': 0});
   
    pred_data_1 = data_1[col[1]].map({'>50K': 1, '<=50K': 0});
    pred_data_2 = data_2[col[3]].map({'>50K': 1, '<=50K': 0});
    
    wskazniki_1 = ocen_model_klasyfikacji_binarnej(true_data, pred_data_1);
    wskazniki_2 = ocen_model_klasyfikacji_binarnej(true_data, pred_data_2);
    
    ROC(true_data, data_1[col[2]], col[2]);
    ROC(true_data, data_2[col[4]], col[4]);
    
    
    print("\nwskazniki_1: ");
    print(wskazniki_1);
    print("wskazniki_2: ");
    print(wskazniki_2);
    resultList = [];
    
    for i in range(len(wskazniki_1)):
        resultList.append(col[2] if wskazniki_1[i] > wskazniki_2[i] else col[4]);
    
    result = ('Lepszy model:' +
    '\nTrafność: ' + resultList[0] +
    '\nCzułość: ' + resultList[1] +
    '\nSpecyficzność: ' + resultList[2] +
    '\nPrecyzja: ' + resultList[3] +
    '\nWynik F1: ' + resultList[4]);
    
    return result;
    
    
def calculate_errors(t_data, p_data):
    tmpMAE = 0;
    tmpMSE = 0;
    tmpMAPE = 0;
    
    n = len(t_data);
    for i in range(n):
        tmpMAE += abs(t_data[i] - p_data[i]);
        tmpMSE += pow((t_data[i] - p_data[i]), 2);
        tmpMAPE += abs(t_data[i] - p_data[i]) / t_data[i];
        
    MAE = tmpMAE/n;
    MSE = tmpMSE/n;
    RMSE = m.sqrt(MSE);
    MAPE = (100/n) * tmpMAPE;
    return MAE, MSE, RMSE, MAPE;

def create_hist(true_data, pred_data):
    global columnNum;
    roznica = [];
    
    for i in range(len(true_data)):
        roznica.append(abs(true_data[i] - pred_data)[i]);
    
    roznica.sort()
    fig = Figure(figsize=(4,4));
    hist1 = fig.add_subplot();
    hist1.hist(roznica, bins=50, color='navy', density=True);
    hist1.plot(roznica, norm.pdf(roznica, st.mean(roznica), st.stdev(roznica)), color='darkorange');
    
    canvas = FigureCanvasTkAgg(fig, master = window);
    canvas.draw();
    canvas.get_tk_widget().grid(column = columnNum, columnspan = 40, row = 2, padx = 5);
    columnNum += 51;
    
    
def regresja(path):
    data = pd.read_csv(path);
    col = data.columns;
        
    data = data.sort_values(by = col[0], ascending=True, ignore_index=True)
    
    true_data = data[col[0]];
    pred_data_1 = data[col[1]];
    pred_data_2 = data[col[2]];
    
    create_hist(true_data, pred_data_1);
    create_hist(true_data, pred_data_2);
    
    MAE_1, MSE_1, RMSE_1, MAPE_1 = calculate_errors(true_data, pred_data_1);
    MAE_2, MSE_2, RMSE_2, MAPE_2 = calculate_errors(true_data, pred_data_2);
    
    result = ('Średni błąd bezwzględny: ' + str(round(MAE_1, 3)) +
            '\nBłąd średniokwadratowy: ' + str(round(MSE_1, 3)) + 
            '\nPierwiastek błędu średniokwadratowego:' + str(round(RMSE_1, 3)) +
            '\nŚredni bezwzględny błąd procentowy:' + str(round(MAPE_1, 3)) + '%' +
            "\n------------------------------------------------" +
            '\nŚredni błąd bezwzględny:' + str(round(MAE_2, 3)) +
            '\nBłąd średniokwadratowy:' + str(round(MSE_2, 3)) +
            '\nPierwiastek błędu średniokwadratowego:' + str(round(RMSE_2, 3)) +
            '\nŚredni bezwzględny błąd procentowy:' + str(round(MAPE_2, 3)) + '%' )
    return result;


columnNum = 0;
path = "";

#pozyskiwanie scieżki pliku
def getFile():
    global path;
    path = fd.askopenfilename(filetypes=[("Excel files", "*.csv")]);


#uruchamianie głównego programu    
def start():
    global columnNum;
    columnNum = 0;
    modelVar = variable.get();
    global path;
    
    #wyswietla okno błędu jesli plik nie został wybrany
    if path == "":
        tk.messagebox.showerror('Error', 'Error: Plik nie wybrany');
        text_box = tk.Text(window, width = 50, height = 10);
        return;

    if modelVar == "Klasyfikacyjny":
        #text_box.destroy()
        result = klasyfikacja_binarna(path);
        text_box = tk.Text(window, width = 50, height = 10);
        text_box.grid(column = 0, columnspan = 40, row = 4, padx = 5, pady=5);
        
        text_box.delete('1.0', "end-1c");
        text_box.insert("end-1c", result);
        path = "";
        
    elif modelVar == "Regresyjny":
        result = regresja(path);
        
        text_box = tk.Text(window, width = 50, height = 10);
        text_box.grid(column = 0, columnspan = 40, row = 4, padx = 5, pady=5);
        
        text_box.delete('1.0', "end-1c");
        text_box.insert("end-1c", result);
        
        path = "";
        
    else:
        #wyswietla okno błędu jesli model nie został wybrany
        tk.messagebox.showerror('Error', 'Error: Model nie wybrany');
        return;
        
        


def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy();
        print("closed");
        sys.exit()


#stwórz okno  
window = tk.Tk();
window.title("Porównanie");
window.geometry("1000x800");

#wybieranie pliku
select_file_button = tk.Button(window, text="Wybierz plik", command=getFile);
select_file_button.grid(column = 0, row = 0);

#wybieranie modelu
variable = tk.StringVar(window, value="Wybierz model");
select_mode = tk.OptionMenu(window, variable, "Klasyfikacyjny", "Regresyjny");
select_mode.config(width=14);
select_mode.grid(column = 1, row = 0);

#start
startButton = tk.Button(window, text="OK", command=start);
startButton.grid(column = 2, row = 0);




window.protocol("WM_DELETE_WINDOW", on_closing);

window.mainloop();


















