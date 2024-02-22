# -*- coding: utf-8 -*-
"""Project module for attitude determination and control system.

This module is simply the control script that utilizes the simulation engine
and plots the results.
"""

from tkinter.messagebox import YES
import glob
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import pandas as pd 
import json, os
import shutil 
from pathlib import Path
from csv import writer
from spacecraft import Spacecraft
from reactionwheels import ReactionWheels
from sensors import Gyros, Magnetometer, EarthHorizonSensor
from controller import PDController
from math_utils import (quaternion_multiply, t1_matrix, t2_matrix, t3_matrix,
                        dcm_to_quaternion, quaternion_to_dcm, normalize, cross)
from simulation import simulate_adcs

def main():

    
    J = 1 / 12 * sc_mass * np.diag([
        sc_dim[1]**2 + sc_dim[2]**2, sc_dim[0]**2 + sc_dim[2]**2,
        sc_dim[0]**2 + sc_dim[1]**2
    ])
    sc_dipole = np.array([0, 0.018, 0])

    # Define two `PDController` objects—one to represent no control and one
    # to represent PD control with the specified gains
    no_controller = PDController(
        k_d=np.diag([0, 0, 0]), k_p=np.diag([0, 0, 0]))
    controller = PDController(
        k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1]))

    # Northrop Grumman LN-200S Gyros
    gyros = Gyros(bias_stability=1, angular_random_walk=0.07)
    perfect_gyros = Gyros(bias_stability=0, angular_random_walk=0)

    # NewSpace Systems Magnetometer
    magnetometer = Magnetometer(resolution=10e-9)
    perfect_magnetometer = Magnetometer(resolution=0)

    # Adcole Maryland Aerospace MAI-SES Static Earth Sensor
    earth_horizon_sensor = EarthHorizonSensor(accuracy=0.25)
    perfect_earth_horizon_sensor = EarthHorizonSensor(accuracy=0)

    # Sinclair Interplanetary 60 mNm-sec RXWLs
    actuators = ReactionWheels(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=20e-3,
        rxwl_max_momentum=0.18,
        noise_factor=0.03)
    perfect_actuators = ReactionWheels(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=np.inf,
        rxwl_max_momentum=np.inf,
        noise_factor=0.0)

    # define some orbital parameters
    mu_earth = 3.986004418e14
    R_e = 6.3781e6
    orbit_w = np.sqrt(mu_earth / orbit_radius**3)
    period = 2 * np.pi / orbit_w

    # define a function that returns the inertial position and velocity of the
    # spacecraft (in m & m/s) at any given time
    def position_velocity_func(t):
        r = orbit_radius / np.sqrt(2) * np.array([
            -np.cos(orbit_w * t),
            np.sqrt(2) * np.sin(orbit_w * t),
            np.cos(orbit_w * t),
        ])
        v = orbit_w * orbit_radius / np.sqrt(2) * np.array([
            np.sin(orbit_w * t),
            np.sqrt(2) * np.cos(orbit_w * t),
            -np.sin(orbit_w * t),
        ])
        return r, v

    # compute the initial inertial position and velocity
    r_0, v_0 = position_velocity_func(0)

    # define the body axes in relation to where we want them to be:
    # x = Earth-pointing
    # y = pointing along the velocity vector
    # z = normal to the orbital plane
    b_x = -normalize(r_0)
    b_y = normalize(v_0)
    b_z = cross(b_x, b_y)

    # construct the nominal DCM from inertial to body (at time 0) from the body
    # axes and compute the equivalent quaternion
    dcm_0_nominal = np.stack([b_x, b_y, b_z])
    q_0_nominal = dcm_to_quaternion(dcm_0_nominal)

    # compute the nominal angular velocity required to achieve the reference
    # attitude; first in inertial coordinates then body
    w_nominal_i = 2 * np.pi / period * normalize(cross(r_0, v_0))
    w_nominal = np.matmul(dcm_0_nominal, w_nominal_i)

    # provide some initial offset in both the attitude and angular velocity
    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.005, 0, 0])

    # define a function that will model perturbations
    def perturb_func(satellite):
        return (satellite.approximate_gravity_gradient_torque() +
                satellite.approximate_magnetic_field_torque())

    # define a function that returns the desired state at any given point in
    # time (the initial state and a subsequent rotation about the body x, y, or
    # z axis depending upon which nominal angular velocity term is nonzero)
    def desired_state_func(t):
        if w_nominal[0] != 0:
            dcm_nominal = np.matmul(t1_matrix(w_nominal[0] * t), dcm_0_nominal)
        elif w_nominal[1] != 0:
            dcm_nominal = np.matmul(t2_matrix(w_nominal[1] * t), dcm_0_nominal)
        elif w_nominal[2] != 0:
            dcm_nominal = np.matmul(t3_matrix(w_nominal[2] * t), dcm_0_nominal)
        return dcm_nominal, w_nominal

    # construct three `Spacecraft` objects composed of all relevant spacecraft
    # parameters and objects that resemble subsystems on-board
    # 1st Spacecraft: no controller
    # 2nd Spacecraft: PD controller with perfect sensors and actuators
    # 3rd Spacecraft: PD controller with imperfect sensors and actuators

    satellite_no_control = Spacecraft(
        J=J,
        controller=no_controller,
        gyros=perfect_gyros,
        magnetometer=perfect_magnetometer,
        earth_horizon_sensor=perfect_earth_horizon_sensor,
        actuators=perfect_actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    satellite_present = Spacecraft(
        J=J,
        controller=controller,
        gyros= present_gyros,
        magnetometer=present_magnetometer,
        earth_horizon_sensor=perfect_earth_horizon_sensor,
        actuators=perfect_actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    satellite_perfect = Spacecraft(
        J=J,
        controller=controller,
        gyros= perfect_gyros,
        magnetometer=perfect_magnetometer,
        earth_horizon_sensor=perfect_earth_horizon_sensor,
        actuators=perfect_actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    # Simulate the behavior of all three spacecraft over time
    simulate(
        satellite=satellite_no_control,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time= stop_time,
        tag=r"(No Control)")

    simulate(
        satellite=satellite_present,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time= stop_time,
        tag=r"(Actual Estimation \& Control)")

    simulate(
        satellite=satellite_perfect,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time= stop_time,
        tag=r"(Perfect Estimation \& Control)")

def simulate(satellite,
             nominal_state_func,
             perturbations_func,
             position_velocity_func,
             stop_time,
             tag=""):

    # carry out the actual simulation and gather the results
    results = simulate_adcs(
        satellite=satellite,
        nominal_state_func=nominal_state_func,
        perturbations_func=perturbations_func,
        position_velocity_func=position_velocity_func,
        start_time=0,
        delta_t=1,
        stop_time=stop_time,
        verbose=True)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.rc("figure", dpi=120)
    plt.rc("savefig", dpi=120)

    # plot the desired results (logged at each delta_t)

    plt.figure(1)
    plt.subplot(411)
    plt.title(r"Evolution of Quaternion Components over Time {}".format(tag))
    plt.plot(results["times"], results["q_actual"][:, 0])
    plt.ylabel(r"$Q_0$")
    plt.subplot(412)
    plt.plot(results["times"], results["q_actual"][:, 1])
    plt.ylabel(r"$Q_1$")
    plt.subplot(413)
    plt.plot(results["times"], results["q_actual"][:, 2])
    plt.ylabel(r"$Q_2$")
    plt.subplot(414)
    plt.plot(results["times"], results["q_actual"][:, 3])
    plt.ylabel(r"$Q_3$")
    plt.xlabel(r"Time (s)")
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(2)
    plt.subplot(311)
    plt.title(r"Evolution of Angular Velocity over Time {}".format(tag))
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(
        results["times"],
        results["w_desired"][:, 0],
        label="desired",
        linestyle="--")
    plt.ylabel(r"$\omega_x$ (rad/s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(
        results["times"],
        results["w_desired"][:, 1],
        label="desired",
        linestyle="--")
    plt.ylabel(r"$\omega_y$ (rad/s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(
        results["times"],
        results["w_desired"][:, 2],
        label="desired",
        linestyle="--")
    plt.ylabel(r"$\omega_z$ (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(3)
    plt.subplot(311)
    plt.title(r"Angular Velocity of Reaction Wheels over Time {}".format(tag))
    plt.plot(results["times"], results["w_rxwls"][:, 0])
    plt.ylabel(r"$\omega_1$ (rad/s)")
    plt.subplot(312)
    plt.plot(results["times"], results["w_rxwls"][:, 1])
    plt.ylabel(r"$\omega_2$ (rad/s)")
    plt.subplot(313)
    plt.plot(results["times"], results["w_rxwls"][:, 2])
    plt.ylabel(r"$\omega_3$ (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(4)
    plt.subplot(311)
    plt.title(r"Perturbation Torques over Time {}".format(tag))
    plt.plot(results["times"], results["M_perturb"][:, 0])
    plt.ylabel(r"$M_x (N \cdot m)$")
    plt.subplot(312)
    plt.plot(results["times"], results["M_perturb"][:, 1])
    plt.ylabel(r"$M_y (N \cdot m)$")
    plt.subplot(313)
    plt.plot(results["times"], results["M_perturb"][:, 2])
    plt.ylabel(r"$M_z (N \cdot m)$")
    plt.xlabel(r"Time (s)")
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(5)
    DCM_actual = np.empty(results["DCM_desired"].shape)
    for i, q in enumerate(results["q_actual"]):
        DCM_actual[i] = quaternion_to_dcm(q)

    k = 1
    for i in range(3):
        for j in range(3):
            plot_num = int("33{}".format(k))
            plt.subplot(plot_num)
            if k == 2:
                plt.title(
                    r"Evolution of DCM Components over Time {}".format(tag))
            plt.plot(results["times"], DCM_actual[:, i, j], label="actual")
            plt.plot(
                results["times"],
                results["DCM_desired"][:, i, j],
                label="desired",
                linestyle="--")
            element = "T_{" + str(i + 1) + str(j + 1) + "}"
            plt.ylabel("$" + element + "$")
            if k >= 7:
                plt.xlabel(r"Time (s)")
            plt.legend()
            k += 1
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.25, wspace=0.3)

    plt.figure(6)
    k = 1
    for i in range(3):
        for j in range(3):
            plot_num = int("33{}".format(k))
            plt.subplot(plot_num)
            if k == 2:
                plt.title(
                    r"Actual vs Estimated Attitude over Time {}".format(tag))
            plt.plot(results["times"], DCM_actual[:, i, j], label="actual")
            plt.plot(
                results["times"],
                results["DCM_estimated"][:, i, j],
                label="estimated",
                linestyle="--",
                color="y")
            element = "T_{" + str(i + 1) + str(j + 1) + "}"
            plt.ylabel("$" + element + "$")
            if k >= 7:
                plt.xlabel(r"Time (s)")
            plt.legend()
            k += 1
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.25, wspace=0.3)

    plt.figure(7)
    plt.subplot(311)
    plt.title(r"Actual vs Estimated Angular Velocity over Time {}".format(tag))
    plt.plot(results["times"], results["w_actual"][:, 0], label="actual")
    plt.plot(
        results["times"],
        results["w_estimated"][:, 0],
        label="estimated",
        linestyle="--",
        color="y")
    plt.ylabel(r"$\omega_x$ (rad/s)")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["w_actual"][:, 1], label="actual")
    plt.plot(
        results["times"],
        results["w_estimated"][:, 1],
        label="estimated",
        linestyle="--",
        color="y")
    plt.ylabel(r"$\omega_y$ (rad/s)")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["w_actual"][:, 2], label="actual")
    plt.plot(
        results["times"],
        results["w_estimated"][:, 2],
        label="estimated",
        linestyle="--",
        color="y")
    plt.ylabel(r"$\omega_z$ (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.figure(8)
    plt.subplot(311)
    plt.title(r"Commanded vs Applied Torques over Time {}".format(tag))
    plt.plot(results["times"], results["M_applied"][:, 0], label="applied")
    plt.plot(
        results["times"],
        results["M_ctrl"][:, 0],
        label="commanded",
        linestyle="--")
    plt.ylabel(r"$M_x (N \cdot m)$")
    plt.legend()
    plt.subplot(312)
    plt.plot(results["times"], results["M_applied"][:, 1], label="applied")
    plt.plot(
        results["times"],
        results["M_ctrl"][:, 1],
        label="commanded",
        linestyle="--")
    plt.ylabel(r"$M_y (N \cdot m)$")
    plt.legend()
    plt.subplot(313)
    plt.plot(results["times"], results["M_applied"][:, 2], label="applied")
    plt.plot(
        results["times"],
        results["M_ctrl"][:, 2],
        label="commanded",
        linestyle="--")
    plt.ylabel(r"$M_z (N \cdot m)$")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.94, hspace=0.3)

    plt.show()

sg.theme("DarkAmber")



path_to_json = os.getcwd() + "/satellites"
data_names= []
data_values = []
for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
  with open(path_to_json +'/'+file_name) as json_file:
    data = json.load(json_file)
    data_values.append(list(data.values()))
    data_temp = list(data.values())
    data_names.append(data_temp[0]) 

path_to_magnet = os.getcwd() + "/magnetometers"
magnet_names= []
magnet_values = []
for file_name in [file for file in os.listdir(path_to_magnet) if file.endswith('.json')]:
  with open(path_to_magnet +'/'+file_name) as json_file:
    magnet = json.load(json_file)
    magnet_values.append(list(magnet.values()))
    magnet_temp = list(magnet.values())
    magnet_names.append(magnet_temp[0])

path_to_gyro = os.getcwd() + "/gyroscopes"
gyro_names= []
gyro_values = []
for file_name in [file for file in os.listdir(path_to_gyro) if file.endswith('.json')]:
  with open(path_to_gyro +'/'+file_name) as json_file:
    gyro = json.load(json_file)
    gyro_values.append(list(gyro.values()))
    gyro_temp = list(gyro.values())
    gyro_names.append(gyro_temp[0])

path_to_horizon = os.getcwd() + "/horizonsensors"
horizon_names= []
horizon_values = []
for file_name in [file for file in os.listdir(path_to_horizon) if file.endswith('.json')]:
  with open(path_to_horizon +'/'+file_name) as json_file:
    horizon = json.load(json_file)
    horizon_values.append(list(horizon.values()))
    horizon_temp = list(horizon.values())
    horizon_names.append(horizon_temp[0])

actuatorslist = ['Reaction Wheel', 'Magnetic Torquers', 'Thrusters']
sensorslist = ['Horizon Sensor', 'Sun Sensor','Magnetometer']
sensorsavail = [['Earth Horizon Sensor','Sensor Location','Sensor Orientation']]
sensorsheader = ['Sensor Kind','Sensor Location', 'Sensor Orientation']

sg.set_options(font=("Arial Bold", 14))
a_head= ['Kind', 'Location', 'Orientation']
a_avail = [['Thruster', 23, 78],
        ['Magneto', 21, 66],
        ['Thruster', 22, 60],
        ['Thruster', 20, 75]]

s_head= ['Kind', 'Location', 'Orientation']
s_avail = [['Horizon Sensor', 23, 78],
        ['Sun Sensor', 21, 66],
        ['Magnetometer', 22, 60],
        ['Sun Sensor', 20, 75]]


tbl1 = sg.Table(values=a_avail, headings=a_head,
   auto_size_columns=True,
   display_row_numbers=True,
   justification='center', key='-TABLE-',
   selected_row_colors='red on yellow',
   enable_events=True,
   enable_click_events=True)

tbl2 = sg.Table(values=s_avail, headings=s_head,
   auto_size_columns=True,
   display_row_numbers=True,
   justification='center', key='-TABLE-',
   selected_row_colors='red on yellow',
   enable_events=True,
   enable_click_events=True)

working_directory = os.getcwd()
##print(glob.glob('*satellite.json'))




col_layout_properties = [ [sg.Combo(data_names,enable_events=True, default_value = 'Available Satellites', s=(15,22) , key = '-SATELLITE-')],
          [sg.Text ("Name of Satellite:"), sg.Input(s=(15,1),key ="-NAME-")],
          [sg.Text ("Mass(g):"), sg.Input(s=(15,1),key = "-MASS-")],
          [sg.Text ("Height(m):"), sg.Input(s=(15,1),key = "-HEIGHT-")],
          [sg.Text ("Length(m):"), sg.Input(s=(15,1),key = "-LENGTH-")],
          [sg.Text ("Width(m):"), sg.Input(s=(15,1),key = "-WIDTH-")],
          [sg.Text ("Radius of Orbit(m):"), sg.Input(s=(15,1),key = "-RADIUS-")],
          [sg.Text ("Choose a Magnetometer:"),sg.Combo(magnet_names,enable_events=True, default_value = 'Available Magnetometers', s=(15,22) , key = '-MAGNET-')],
          [sg.Text ("Choose a Horizon Sensor:"),sg.Combo(horizon_names,enable_events=True, default_value = 'Available Horizon Sensors', s=(15,22) , key = '-HORIZON-')],
          [sg.Text ("Choose a Gyroscope:"),sg.Combo(gyro_names,enable_events=True, default_value = 'Available Gyroscopes', s=(15,22) , key = '-GYRO-')],
          [sg.Button("Add Actuator")],
          [sg.Button("Save New Satellite")]
]

col_layout_display = [[sg.Text("Current Actuators on Satellite:"), sg.Button("Display", key = '-ACTUATORSPRES-')],
                      [sg.Text("Current Sensors on Satellite:"),sg.Button("Display", key = '-SENSORSPRES')],
                      [sg.Text("Interactive view of Satellite:"), sg.Button("Display", key = '-3DDISPLAY-')],
                      [sg.Text ("Specify time length of Simulation(sec):"), sg.Input(s=(15,1),key = "-TIMELENGTH-")],
                      [sg.Text("Perform Simulation:"),sg.Button("Simulate")]
                      ]


layout1 = [[sg.Column(col_layout_properties),sg.VSeperator(),sg.Column(col_layout_display,vertical_alignment = 'top')]]

layout2 = [[sg.Text("Load in a JSON file:")],
          [sg.InputText(s =(15,2), key="-FILE_PATH-"), 
           sg.FileBrowse(initial_folder = working_directory, file_types = [("JSON Files", "*.json")]),
           sg.Button("Submit")
          ]]


col_layout_sensor = [[sg.Text("Magnetometers:")],
           [sg.Text("Name:"), sg.Input(s=(15,1),key = "-MAGNETNAME-")],
           [sg.Text("Resolution:"), sg.Input(s=(15,1),key = "-RESOLUTION-")],
           [sg.Button("Add Magnetometer")],
           [sg.Text("Earth Horizon Sensor:")],
           [sg.Text("Name:"), sg.Input(s=(15,1),key = "-HORZNAME-")],
           [sg.Text("Accuracy:"), sg.Input(s=(15,1),key = "-ACCURACY-")], 
           [sg.Button("Add Horizon Sensor")],
           [sg.Text("Gyroscope:")],
           [sg.Text("Name:"), sg.Input(s=(15,1),key = "-GYRONAME-")],
           [sg.Text("Bias Stability:"), sg.Input(s=(15,1),key = "-BIAS-")], 
           [sg.Text("Angular Random Walk:"), sg.Input(s=(15,1),key = "-ANGRANDOM-")], 
           [sg.Button("Add Gyroscope")]
          ]

layout3 = [[sg.Column(col_layout_sensor)]]

layout4 = [[ sg.Text("Actuator kind:"), sg.Combo(actuatorslist,key = "-ActuatorKind-")],
           [sg.Text("Placement location:"), sg.Input(s=(15,1),key = "-ActuatorPlace-")],
          [sg.Text("Placement orientation:"), sg.Input(s=(15,1),key = "-ActuatorOrientation-")],
          [sg.Text("Actuators List:"), sg.Button("Display", key = '-ACTUATORSLIST-')],
          [sg.Button("Add Actuator")]
           ]

tabgrp = [[sg.TabGroup([[sg.Tab('Parameters and Satellite Creation', layout1),
            sg.Tab('Load JSON Configuration', layout2),
            sg.Tab('Sensors', layout3),
            sg.Tab('Actuators', layout4)
            ]])
          ]]

window = sg.Window("ACDS Simulation", tabgrp)

def actuators_list_window():
    layout = [[tbl1]]
    window = sg.Window("Actuator List", layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
          break
        if '+CLICKED+' in event:
          sg.popup("You clicked row:{} Column: {}".format(event[2][0], event[2][1]))
    window.close()

def sensors_list_window():
    layout = [[tbl2]]
    window = sg.Window("Sensor List", layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
          break
        if '+CLICKED+' in event:
          sg.popup("You clicked row:{} Column: {}".format(event[2][0], event[2][1]))
    window.close()

while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, "Exit"):
        break
    if event == "Simulate":
        sc_mass = float(values["-MASS-"])
        sc_dim = [float(values["-HEIGHT-"]),float(values["-LENGTH-"]),float(values["-WIDTH-"])]
        orbit_radius=float(values["-RADIUS-"])
        stop_time = float(values['-TIMELENGTH-'])

        res = magnet_values[magnet_names.index(values['-MAGNET-'])][1]
        resolution = float(res)
        present_magnetometer = Magnetometer(resolution = resolution)

        bstability = gyro_values[gyro_names.index(values['-GYRO-'])][1]
        angrandom_walk = gyro_values[gyro_names.index(values['-GYRO-'])][2]
        bias_stability = float(bstability)
        angular_random_walk = float(angrandom_walk)
        present_gyros = Gyros(bias_stability = bias_stability, angular_random_walk = angular_random_walk)

        acc = horizon_values[horizon_names.index(values['-HORIZON-'])][1]
        accuracy = float(acc)
        present_earth_horizon_sensor = EarthHorizonSensor(accuracy = accuracy)
        if __name__ == "__main__":
            main()
    if event == "Add new":
        new_sat = {
        "Name":values["-NAME-"],
        "Mass":values["-MASS-"],
        "Height":values["-HEIGHT-"],
        "Length":values["-LENGTH-"],
        "Width":values["-WIDTH-"],
        "Radius of Orbit":values["-RADIUS-"]
        } 
        
        json_object = json.dumps(new_sat, indent=4)
        with open(os.path.join(path_to_json,values["-NAME-"]+"-satellite.json"), "w") as outfile:
            outfile.write(json_object)
        for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
               with open(path_to_json +'/'+file_name) as json_file:
                  data = json.load(json_file)
                  data_values.append(list(data.values()))
                  data_temp = list(data.values())
                  data_names.append(data_temp[0])
        window.refresh()
    
    if event == '-SATELLITE-':
        window["-NAME-"].update(values['-SATELLITE-'])
        window["-MASS-"].update(data_values[data_names.index(values['-SATELLITE-'])][1])
        window["-HEIGHT-"].update(data_values[data_names.index(values['-SATELLITE-'])][2]) 
        window["-LENGTH-"].update(data_values[data_names.index(values['-SATELLITE-'])][3])
        window["-WIDTH-"].update(data_values[data_names.index(values['-SATELLITE-'])][4])
        window["-RADIUS-"].update(data_values[data_names.index(values['-SATELLITE-'])][5])
        window['-MAGNET-'].update(data_values[data_names.index(values['-SATELLITE-'])][6])
        window['-HORIZON-'].update(data_values[data_names.index(values['-SATELLITE-'])][7])
        window['-GYRO-'].update(data_values[data_names.index(values['-SATELLITE-'])][8]) 
        window.refresh()

    if event == "Submit":
        address = values["-FILE_PATH-"]
        shutil.move(address, path_to_json)

    if event == '-3DDISPLAY-':
       from vpython import *
       box(
       size = vector(float(values["-HEIGHT-"]),float(values["-LENGTH-"]),float(values["-WIDTH-"])),
        opacity = 0.5
       )
       cylinder(pos = vector(0.05,0,0),
       length = 0.01,
       radius = 0.01,
       axis = vector(1,0,0),
       colour = vector(1,0,0)
       )
       cylinder(pos = vector(0,0.113,0),
       length = 0.01,
       radius = 0.01,
       axis = vector(0,1,0),
       colour = vector(1,0,0)
       )      
       cylinder(pos = vector(0,0,0.1825),
       length = 0.01,
       radius = 0.01,
       axis = vector(0,0,1),
       colour = (1,0,0)
       )             
       scene.caption= """
        To rotate "camera", drag with right button or Ctrl-drag.
        To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
        On a two-button mouse, middle is left + right.
        To pan left/right and up/down, Shift-drag.sss
        """

    if event == '-ACTUATORSLIST-':
        actuators_list_window()
       
    if event == '-SENSORSLIST-':
        sensors_list_window()
    if event == 'Add Actuator':
       a_avail.append([values["-ActuatorKind-"],values["-ActuatorPlace-"],values["ActuatorOrientation"]])

    if event ==  'Add Magnetometer':
        new_mag = {
        "Name":values["-MAGNETNAME-"],
        "Resolution":values["-RESOLUTION-"]
        } 
        
        json_object = json.dumps(new_mag, indent=4)
        with open(os.path.join(path_to_magnet,values["-MAGNETNAME-"]+".json"), "w") as outfile:
            outfile.write(json_object)
        for file_name in [file for file in os.listdir(path_to_magnet) if file.endswith('.json')]:
               with open(path_to_magnet +'/'+file_name) as json_file:
                    magnet = json.load(json_file)
                    magnet_values.append(list(magnet.values()))
                    magnet_temp = list(magnet.values())
                    magnet_names.append(magnet_temp[0])
        window.refresh()


    
window.close()




    