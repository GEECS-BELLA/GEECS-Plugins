import requests
from hszinc import parse
import hszinc

def read_file(file_name):
    try:
        with open(file_name, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def parse_zinc_output(zinc_output):
    try:
        grid = hszinc.parse(zinc_output)
        return grid
    except Exception as e:
        print(f"Failed to parse Zinc output: {e}")
        return None        
def main():
    file_name = "datfile.txt"  # Update with your file name
    
    # Read the text file
    zinc_output = read_file(file_name)
    if zinc_output is None:
        return
    
    print("Zinc Output:")
    print(zinc_output)

    # Parse the Zinc output
    grid = parse_zinc_output(zinc_output)
    if grid is None:
        return
    
    print("\nParsed Grid:")
    for row in grid:
        print(row)

    export_to_csv(grid, 'parsedfile.csv')

    plot_data(grid)

import csv

def export_to_csv(grid, csv_file_name):
    try:
        with open(csv_file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header
            header = grid.column.keys()
            writer.writerow(header)
            
            # Write the rows
            for row in grid:
                writer.writerow([row.get(col, '') for col in header])
                
        print(f"Data successfully exported to {csv_file_name}")
    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")

import matplotlib.pyplot as plt

def plot_data(grid):
    try:
        # Assuming the grid has columns 'time' and 'value'
        times = [row['ts'] for row in grid]
        values = [row['v0'] for row in grid]

        plt.figure(figsize=(10, 5))
        plt.plot(times, values, marker='o', linestyle='-', color='b')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Line Graph of Parsed Data')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the data: {e}")

if __name__ == "__main__":
    main()