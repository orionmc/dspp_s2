import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the path to CSV file
csv_file_path = 'C:\\Users\\user0\\Documents\\BPP\\DS-Proffessional-Practice\\_DataSets\\Heart-Attack\\heart0.csv'

# Open the CSV file
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

   # Find the maximum width of each column
    column_widths = [max(len(cell) for cell in col) for col in zip(*csv_reader)]
    
    # Position to the beginning of the file and read again 
    file.seek(0)
    csv_reader = csv.reader(file)
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Format each cell in the row with appropriate width
        formatted_row = [cell.ljust(width) for cell, width in zip(row, column_widths)]
        # Join the formatted cells with a separator (space) and print
        print('  '.join(formatted_row))
    

    
    #null_count = csv_reader.isnull().sum()
    #print(null_count)

