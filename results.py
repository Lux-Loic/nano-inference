# ----------------------------------------
# Imports
# ----------------------------------------
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
from PIL import Image
import sys
 

if __name__ == "__main__":
    # ----------------------------------------
    # Save Results
    # ----------------------------------------
    results_path = "./results"
    # Create results folder if it doesn't exist

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plots_path = os.path.join(results_path, "./plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # ----------------------------------------
    # CSV
    # ----------------------------------------
    logs = open('stats.log', 'r')
    lines = logs.read().splitlines()
    logs.close()
    rows = []

    for line in lines:
        rows.append(json.loads(line.replace("\'", "\"")))
        
    df = pd.DataFrame(rows)

    # ----------------------------------------
    # Plots
    # ----------------------------------------
    # CPUs
    CPU1 = df["CPU1"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('CPU1')
    ax.set_xlabel('Time')
    ax.set_ylabel('CPU Usage (%)')
    ax.plot(np.arange(len(CPU1)), CPU1)
    fig.savefig(plots_path + '/CPU1.png')
    plt.close(fig)
    CPU2 = df["CPU2"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('CPU2')
    ax.set_xlabel('Time')
    ax.set_ylabel('CPU2 Usage (%)')
    ax.plot(np.arange(len(CPU2)), CPU2)
    fig.savefig(plots_path + '/CPU2.png')
    plt.close(fig)
    CPU3 = df["CPU3"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('CPU3')
    ax.set_xlabel('Time')
    ax.set_ylabel('CPU3 Usage (%)')
    ax.plot(np.arange(len(CPU3)), CPU3)
    fig.savefig(plots_path + '/CPU3.png')
    plt.close(fig)
    CPU4 = df["CPU4"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('CPU4')
    ax.set_xlabel('Time')
    ax.set_ylabel('CPU4 Usage (%)')
    ax.plot(np.arange(len(CPU4)), CPU4)
    fig.savefig(plots_path + '/CPU4.png')
    plt.close(fig)

    # GPU
    GPU = df["GPU"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('GPU Progression')
    ax.set_xlabel('Time')
    ax.set_ylabel('GPU Usage (%)')
    ax.plot(np.arange(len(GPU)), GPU)
    fig.savefig(plots_path + '/GPU.png')
    plt.close(fig)

    # RAM
    RAM = df["RAM"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('RAM Progression')
    ax.set_xlabel('Time')
    ax.set_ylabel('RAM Usage (kB)')
    ax.plot(np.arange(len(RAM)), RAM)
    fig.savefig(plots_path + '/RAM.png')
    plt.close(fig)

    # Temperatures
    temp_ao = df["Temp AO"]
    temp_cpu = df["Temp CPU"]
    temp_gpu = df["Temp GPU"]
    temp_pll = df["Temp PLL"]
    temp_thermal = df["Temp thermal"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('AO Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (Celcius)')
    ax.plot(np.arange(len(temp_ao)), temp_ao)
    fig.savefig(plots_path + '/temp_ao.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('CPU Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (Celcius)')
    ax.plot(np.arange(len(temp_cpu)), temp_cpu)
    fig.savefig(plots_path + '/temp_cpu.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('GPU Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (Celcius)')
    ax.plot(np.arange(len(temp_gpu)), temp_gpu)
    fig.savefig(plots_path + '/temp_gpu.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('PLL Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (Celcius)')
    ax.plot(np.arange(len(temp_pll)), temp_pll)
    fig.savefig(plots_path + '/temp_pll.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Thermal Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (Celcius)')
    ax.plot(np.arange(len(temp_thermal)), temp_thermal)
    fig.savefig(plots_path + '/temp_thermal.png')
    plt.close(fig)

    # Current Power
    power_cur = df["power cur"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Current Power')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Consumption (Milliwatt)')
    ax.plot(np.arange(len(power_cur)), power_cur)
    fig.savefig(plots_path + '/power_cur.png')
    plt.close(fig)

    # Average Power
    power_avg = df["power avg"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Average Power')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Consumption (Milliwatt)')
    ax.plot(np.arange(len(power_avg)), power_avg)
    fig.savefig(plots_path + '/power_avg.png')
    plt.close(fig)

    # ----------------------------------------
    # Mean stats
    # ----------------------------------------
    numerical_df = df[[
        "CPU1",
        "CPU2",
        "CPU3",
        "CPU4",
        "GPU",
        "RAM",
        "EMC",
        "IRAM",
        "SWAP",
        "APE",
        "Temp AO",
        "Temp CPU",
        "Temp GPU",
        "Temp PLL",
        "Temp thermal",
        "power cur",
        "power avg"
    ]]
    means_df = numerical_df.mean(axis = 0)
    means_df.to_csv(os.path.join(results_path, "mean.csv"), sep=',')













