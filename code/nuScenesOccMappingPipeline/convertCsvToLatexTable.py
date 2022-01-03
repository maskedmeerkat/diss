print("# Import Libraries")
import os
import sys
import time
import csv
import numpy as np
from tqdm import tqdm

# get all csv files
logDir = "./models/exp_deep_ism_comparison/"
csvFileNames = [csvFileName for csvFileName in os.listdir(logDir) if csvFileName.endswith(".csv")]
csvFileNames = ["ilm_scores.csv"]

# read each csv file
for csvFileName in tqdm(csvFileNames):
    with open(logDir+csvFileName) as csvFile, open(logDir+csvFileName[:-len("_scores.csv")]+"_latex.txt", 'w') as txtFile:                
        csvReader = csv.reader(csvFile, delimiter=',')
        confMat = np.zeros((4,12))
        
        # retrieve data from csv file
        confMatRow = 0
        confMatCol = 0
        for iRow, row in enumerate(csvReader):
            # print(row[1:])
            if (((iRow >  6) and (iRow < 11)) or 
                ((iRow > 12) and (iRow < 17)) or
                ((iRow > 18) and (iRow < 23))):
                confMat[confMatRow,confMatCol:confMatCol+4] = np.array(row[1:], dtype=float).round(1)
                confMatRow += 1
            if (confMatRow >= 4):
                confMatCol += 4
                confMatRow = 0
        
        # write txt file as latex table
        confMat = confMat.astype(str)
        confMat[confMat == "-1.0"] = "-"
        txtFile.write('\\begin{tabular}{c|c|cccc|cccc|cccc}\n')		
        txtFile.write("&$k$ & $d$ & $f$ & $o$ & $u$ & $d$ & $f$ & $o$ & $u$ & $d$ & $f$ & $o$ & $u$\\\\\n")
        txtFile.write("\\hline\n")  
        for iRow in range(confMat.shape[0]):
            # write row identifier
            if (iRow == 0):
                txtFile.write(("\\parbox[t]{2mm}{\multirow{4}{*}{\\rotatebox[origin=c]{90}{ShiftNet}}}&$p(k|d)$ & \\textcolor{mygreen}{" +confMat[iRow,0]+ "}" +
                               " & \\textcolor{myred}{" +confMat[iRow,1]+ "}"+
                               " & \\textcolor{myred}{" +confMat[iRow,2]+ "}"+
                               " & " +confMat[iRow,3]+
                               " & \\textcolor{mygreen}{" +confMat[iRow,4]+ "}"
                               " & \\textcolor{myred}{" +confMat[iRow,5]+ "}"+
                               " & \\textcolor{myred}{" +confMat[iRow,6]+ "}"+
                               " & " +confMat[iRow,7]+
                               " & \\textcolor{mygreen}{" +confMat[iRow,8]+ "}"
                               " & \\textcolor{myred}{" +confMat[iRow,9]+ "}"+
                               " & \\textcolor{myred}{" +confMat[iRow,10]+ "}"+
                               " & " +confMat[iRow,11]+ "\\\\\n"))
            elif (iRow == 1):
                txtFile.write(("&$p(k|f)$ & \\textcolor{myred}{" +confMat[iRow,0]+ "}"+
                               " & \\textcolor{mygreen}{" +confMat[iRow,1]+ "}"
                               " & \\textcolor{myred}{" +confMat[iRow,2]+ "}"+
                               " & " +confMat[iRow,3]+
                               " & \\textcolor{myred}{" +confMat[iRow,4]+ "}"+
                               " & \\textcolor{mygreen}{" +confMat[iRow,5]+ "}"
                               " & \\textcolor{myred}{" +confMat[iRow,6]+ "}"+
                               " & " +confMat[iRow,7]+
                               " & \\textcolor{myred}{" +confMat[iRow,8]+ "}"+
                               " & \\textcolor{mygreen}{" +confMat[iRow,9]+ "}"
                               " & \\textcolor{myred}{" +confMat[iRow,10]+ "}"+
                               " & " +confMat[iRow,11]+ "\\\\\n"))
            elif (iRow == 2):
                txtFile.write(("&$p(k|o)$ & \\textcolor{myred}{" +confMat[iRow,0]+ "}"+
                               " & \\textcolor{myred}{" +confMat[iRow,1]+ "}"+
                               " & \\textcolor{mygreen}{" +confMat[iRow,2]+ "}"
                               " & " +confMat[iRow,3]+
                               " & \\textcolor{myred}{" +confMat[iRow,4]+ "}"+
                               " & \\textcolor{myred}{" +confMat[iRow,5]+ "}"+
                               " & \\textcolor{mygreen}{" +confMat[iRow,6]+ "}"                               
                               " & " +confMat[iRow,7]+
                               " & \\textcolor{myred}{" +confMat[iRow,8]+ "}"+
                               " & \\textcolor{myred}{" +confMat[iRow,9]+ "}"+
                               " & \\textcolor{mygreen}{" +confMat[iRow,10]+ "}"
                               " & " +confMat[iRow,11]+ "\\\\\n"))
            elif (iRow == 3):
                txtFile.write(("&$p(k|u)$ & "+confMat[iRow,0]+
                               " & " +confMat[iRow,1]+
                               " & " +confMat[iRow,2]+ 
                               " & " +confMat[iRow,3]+                              
                               " & " +confMat[iRow,4]+
                               " & " +confMat[iRow,5]+
                               " & " +confMat[iRow,6]+
                               " & " +confMat[iRow,7]+                                                             
                               " & " +confMat[iRow,8]+ 
                               " & " +confMat[iRow,9]+ 
                               " & " +confMat[iRow,10]+ 
                               " & " +confMat[iRow,11]+ "\\\\\n"))
        txtFile.write("\\hline\n")
        txtFile.write("&& \\multicolumn{4}{c|}{overall} & \\multicolumn{4}{c|}{visible} & \\multicolumn{4}{c}{occluded}\n")
        txtFile.write("\\end{tabular}\n")
        
        csvFile.close()
        txtFile.close()
        
        
            
    
        
        
        
        
        
        
        
        
        
        
    